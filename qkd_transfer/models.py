"""Small binary models. Learned adapters are shared and travel with the encoder."""
import copy
import random
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight
from .features import summaries


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Network(nn.Module):
    def __init__(self, n_features, kind, representation='combined', latent=8):
        super().__init__()
        self.kind, self.representation = kind, representation
        self.encoder = nn.Sequential(nn.Linear(2*n_features, 32), nn.ReLU(), nn.Linear(32, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, n_features))
        self.predictor = nn.GRU(latent, latent, batch_first=True)
        learned = kind in ('ae_lstm', 'jepa_lstm')
        width = latent if learned and representation == 'latent' else 2*n_features
        if learned and representation == 'combined':
            width += latent
        self.window = nn.Sequential(nn.Linear(width, 32), nn.ReLU())
        self.temporal = nn.LSTM(32, 32, batch_first=True) if kind.endswith('lstm') else nn.Identity()
        self.head = nn.Linear(32, 1)

    def latent(self, x, mask):
        return self.encoder(torch.cat([x, mask.float()], -1))

    def embedding(self, x, mask):
        raw = torch.cat([x, mask.float()], -1)
        if self.kind in ('ae_lstm', 'jepa_lstm'):
            z = self.latent(x, mask)
            raw = z if self.representation == 'latent' else torch.cat([raw, z], -1)
        h = self.window(raw)
        if isinstance(self.temporal, nn.LSTM):
            h, _ = self.temporal(h)
        valid = mask.any(-1).float()
        return (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)

    def forward(self, x, mask):
        return self.head(self.embedding(x, mask)).squeeze(-1)


def batches(n, size, rng=None):
    ids = np.arange(n)
    if rng is not None:
        rng.shuffle(ids)
    for start in range(0, n, size):
        yield ids[start:start+size]


def tensor(a, device):
    return torch.as_tensor(a, device=device)


def pretrain(net, x, m, vx, vm, args, seed):
    if not len(x) or not len(vx):
        raise ValueError('Honest train and validation runs required for representation learning')
    device = args.device
    net.to(device)
    teacher = copy.deepcopy(net.encoder).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(list(net.encoder.parameters()) + list(net.decoder.parameters()) + list(net.predictor.parameters()), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history, best, stale = [], float('inf'), 0
    best_state = copy.deepcopy(net.state_dict())

    def objective(a, mask, training):
        z = net.latent(a, mask)
        if net.kind == 'ae_lstm':
            residual = (net.decoder(z)-a).square() * mask
            per_run = residual.sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
            return per_run.mean()
        # Predict masked teacher windows using a causal GRU over visible online latents.
        if training:
            hidden = torch.rand(z.shape[:2], device=device) < .25
        else:
            hidden = torch.zeros(z.shape[:2], dtype=torch.bool, device=device)
            hidden[:, 1::4] = True
        hidden[:, 0] = False
        hidden[:, -1] = True
        predicted, _ = net.predictor(z.masked_fill(hidden[..., None], 0))
        with torch.no_grad():
            target = teacher(torch.cat([a, mask.float()], -1))
        usable = hidden & mask.any(-1)
        loss = ((predicted-target).square().mean(-1)*usable).sum()/usable.sum().clamp_min(1)
        visible_z = z[mask.any(-1)]
        variance = torch.relu(1-torch.sqrt(visible_z.var(0, unbiased=False)+1e-4)).mean() if len(visible_z) else z.sum()*0
        return loss + .1*variance

    for epoch in range(args.epochs):
        net.train()
        for ids in batches(len(x), args.batch_size, rng):
            optimizer.zero_grad()
            loss = objective(tensor(x[ids], device), tensor(m[ids], device), True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                for t, p in zip(teacher.parameters(), net.encoder.parameters()):
                    t.mul_(.99).add_(p, alpha=.01)
        net.eval()
        total = 0.
        with torch.no_grad():
            for ids in batches(len(vx), args.batch_size):
                total += float(objective(tensor(vx[ids], device), tensor(vm[ids], device), False))*len(ids)
        value = total/len(vx)
        history.append({'epoch': epoch+1, 'honest_validation_loss': value})
        if epoch == 0 or (epoch+1) % 10 == 0:
            print('  honest pretraining epoch', epoch+1, 'validation loss', round(value, 5), flush=True)
        if value < best-1e-4:
            best, stale, best_state = value, 0, copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    net.load_state_dict(best_state)
    net.cpu()
    return history


class BinaryModel:
    def __init__(self, kind, n_features, representation='combined'):
        self.kind = kind
        self.net = Network(n_features, kind, representation) if kind in ('mlp', 'lstm', 'ae_lstm', 'jepa_lstm') else None
        self.estimator = None
        self.history = {}

    def fit(self, x, m, y, vx, vm, vy, args, seed, mode='scratch'):
        if set(np.unique(y)) != {0, 1} or set(np.unique(vy)) != {0, 1}:
            raise ValueError('Both binary classes required for training and validation')
        seed_all(seed)
        if self.net is None:
            if mode != 'scratch':
                raise ValueError('Head-only/partial adaptation requires a neural model')
            models = {
                'logistic': lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed),
                'rf': lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight='balanced', random_state=seed, n_jobs=args.threads),
                'boosting': lambda: HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=seed),
            }
            if self.kind == 'qber':
                return self
            self.estimator = models[self.kind]()
            kw = {'sample_weight': compute_sample_weight('balanced', y)} if self.kind == 'boosting' else {}
            self.estimator.fit(summaries(x, m), y, **kw)
            return self
        net = self.net
        if mode == 'scratch' and self.kind in ('ae_lstm', 'jepa_lstm'):
            self.history['pretraining'] = pretrain(net, x[y == 0], m[y == 0], vx[vy == 0], vm[vy == 0], args, seed)
        for p in net.parameters():
            p.requires_grad = False
        if mode == 'scratch':
            modules = (net.window, net.temporal, net.head)
        else:
            # Preserve the complete source input mapping; initialise only the target head.
            net.head.reset_parameters()
            modules = (net.head,) if mode == 'head_only' else (net.temporal, net.head)
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True
        self.history['trainable_parameters'] = [n for n, p in net.named_parameters() if p.requires_grad]
        device = args.device
        net.to(device)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
        weights = compute_sample_weight('balanced', y).astype('float32')
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        best, stale, best_state = float('inf'), 0, copy.deepcopy(net.state_dict())
        rng, history = np.random.default_rng(seed), []
        for epoch in range(args.epochs):
            net.train()
            for ids in batches(len(x), args.batch_size, rng):
                optimizer.zero_grad()
                logits = net(tensor(x[ids], device), tensor(m[ids], device))
                loss = (criterion(logits, tensor(y[ids].astype('float32'), device))*tensor(weights[ids], device)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 5)
                optimizer.step()
            score = self.score(vx, vm, args)
            value = log_loss(vy, score, sample_weight=compute_sample_weight('balanced', vy), labels=[0, 1])
            history.append({'epoch': epoch+1, 'validation_log_loss': float(value)})
            if epoch == 0 or (epoch+1) % 10 == 0:
                print('  classifier epoch', epoch+1, 'validation loss', round(float(value), 5), flush=True)
            if value < best-1e-4:
                best, stale, best_state = value, 0, copy.deepcopy(net.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        net.load_state_dict(best_state)
        net.cpu()
        self.history['classification'] = history
        return self

    def score(self, x, m, args):
        if self.net is None:
            if self.kind == 'qber':
                # Runner forces raw qber features: mean window disagreement, no learned model.
                return (np.where(m[..., 0], x[..., 0], 0).sum(1)/m[..., 0].sum(1).clip(1)).clip(0, 1)
            return self.estimator.predict_proba(summaries(x, m))[:, 1]
        self.net.to(args.device).eval()
        output = []
        with torch.no_grad():
            for ids in batches(len(x), args.batch_size):
                output.append(self.net(tensor(x[ids], args.device), tensor(m[ids], args.device)).sigmoid().cpu().numpy())
        return np.concatenate(output)

    def embeddings(self, x, m, args, latent=False):
        if self.net is None:
            return summaries(x, m)
        self.net.to(args.device).eval()
        output = []
        with torch.no_grad():
            for ids in batches(len(x), args.batch_size):
                a, mask = tensor(x[ids], args.device), tensor(m[ids], args.device)
                if latent:
                    z = self.net.latent(a, mask)
                    valid = mask.any(-1).float()
                    h = (z*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
                else:
                    h = self.net.embedding(a, mask)
                output.append(h.cpu().numpy())
        return np.concatenate(output)
