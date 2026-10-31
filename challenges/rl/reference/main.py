"""Reference solution for the reinforcement-learning course. Uses numpy for the neural-net stages."""
import math
import sys

import numpy as np

M64 = (1 << 64) - 1


def fmt(x):
    s = f"{x:.4f}"
    return "0.0000" if s == "-0.0000" else s


def fmts(xs):
    return " ".join(fmt(x) for x in xs)


def lines():
    return [l for l in sys.stdin.read().splitlines() if l.strip()]


def floats(s):
    return [float(t) for t in s.split()]


# ---------- stage 1: splitmix64 ----------

class Rng:
    def __init__(self, seed):
        self.state = seed & M64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uniform(self, lo=0.0, hi=1.0):
        return lo + (hi - lo) * ((self.next() >> 11) / 9007199254740992.0)

    def randint(self, n):
        return int(self.uniform() * n)

    def normal(self):
        u1, u2 = self.uniform(), self.uniform()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def cmd_rng(args):
    r = Rng(int(args[0]))
    n = int(args[1])
    if len(args) > 2:
        k = int(args[2])
        print(" ".join(str(r.randint(k)) for _ in range(n)))
    else:
        print(fmts(r.uniform() for _ in range(n)))


# ---------- stages 2-4: environments ----------

class CartPole:
    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    total_mass = masspole + masscart
    polemass_length = masspole * length
    theta_threshold = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    limit = 500

    def reset(self, rng):
        self.state = [rng.uniform(-0.05, 0.05) for _ in range(4)]
        return list(self.state)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = [x, x_dot, theta, theta_dot]
        terminated = abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
        return list(self.state), 1.0, terminated

    def random_action(self, rng):
        return rng.randint(2)


def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


class Pendulum:
    g, m, l, dt, max_speed, max_torque = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0
    limit = 200

    def obs(self):
        th, thdot = self.state
        return [math.cos(th), math.sin(th), thdot]

    def reset(self, rng):
        self.state = [rng.uniform(-math.pi, math.pi), rng.uniform(-1.0, 1.0)]
        return self.obs()

    def step(self, u):
        th, thdot = self.state
        u = min(max(u, -self.max_torque), self.max_torque)
        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2
        newthdot = thdot + (3 * self.g / (2 * self.l) * math.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = min(max(newthdot, -self.max_speed), self.max_speed)
        newth = th + newthdot * self.dt
        self.state = [newth, newthdot]
        return self.obs(), -cost, False

    def random_action(self, rng):
        return rng.uniform(-self.max_torque, self.max_torque)


def flag(b):
    return "true" if b else "false"


def cmd_cartpole(args):
    ls = lines()
    env = CartPole()
    env.state = floats(ls[0])
    for l in ls[1:]:
        obs, r, term = env.step(int(l))
        print(fmts(obs), fmt(r), flag(term))
        if term:
            break


def cmd_pendulum(args):
    ls = lines()
    env = Pendulum()
    env.state = floats(ls[0])
    for l in ls[1:]:
        obs, r, _ = env.step(float(l))
        print(fmts(obs), fmt(r))


def make_env(name):
    return {"cartpole": CartPole, "pendulum": Pendulum}[name]()


def run_episode(env, rng, policy):
    obs = env.reset(rng)
    length, total = 0, 0.0
    while True:
        obs, r, term = env.step(policy(obs))
        length += 1
        total += r
        if term or length >= env.limit:
            return length, total


def cmd_random_agent(args):
    env, rng, n = make_env(args[0]), Rng(int(args[1])), int(args[2])
    lens, rews = [], []
    for ep in range(n):
        length, total = run_episode(env, rng, lambda obs: env.random_action(rng))
        lens.append(length)
        rews.append(total)
        print(f"Episode {ep} | length: {length}, reward: {fmt(total)}")
    print(f"Average | length: {fmt(sum(lens) / n)}, reward: {fmt(sum(rews) / n)}")


# ---------- stage 5: returns ----------

def cmd_returns(args):
    gamma = float(args[0])
    for l in lines():
        g, out = 0.0, []
        for r in reversed(floats(l)):
            g = r + gamma * g
            out.append(g)
        print(fmts(reversed(out)))


# ---------- stage 6: exploration ----------

def linear_schedule(start, end, fraction, total, t):
    return start + (end - start) * min(1.0, t / (fraction * total))


def argmax(xs):
    best = 0
    for i, x in enumerate(xs):
        if x > xs[best]:
            best = i
    return best


def egreedy(rng, eps, q):
    if rng.uniform() < eps:
        return rng.randint(len(q))
    return argmax(q)


def cmd_schedule(args):
    start, end, fraction, total = map(float, args[:4])
    print(" ".join(fmt(linear_schedule(start, end, fraction, total, float(t))) for t in lines()))


def cmd_egreedy(args):
    eps, rng = float(args[0]), Rng(int(args[1]))
    print(" ".join(str(egreedy(rng, eps, floats(l))) for l in lines()))


# ---------- stages 7-9: tabular Q-learning ----------

BOUNDS = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.5, 3.5)]


def discretize(obs, bins):
    index = 0
    for v, (lo, hi), n in zip(obs, BOUNDS, bins):
        b = min(max(math.floor((v - lo) / (hi - lo) * n), 0), n - 1)
        index = index * n + b
    return index


def cmd_discretize(args):
    bins = [int(a) for a in args[:4]]
    print(" ".join(str(discretize(floats(l), bins)) for l in lines()))


def q_update(q, alpha, gamma, s, a, r, s2, term):
    target = r if term else r + gamma * max(q[s2])
    td = target - q[s][a]
    q[s][a] += alpha * td
    return td


def cmd_qupdate(args):
    alpha, gamma, ns, na = float(args[0]), float(args[1]), int(args[2]), int(args[3])
    q = [[0.0] * na for _ in range(ns)]
    tds = []
    for l in lines():
        s, a, r, s2, term = l.split()
        tds.append(q_update(q, alpha, gamma, int(s), int(a), float(r), int(s2), term == "1"))
    print(fmts(tds))
    for row in q:
        print(fmts(row))


def evaluate(env, rng, policy, episodes):
    return sum(run_episode(env, rng, policy)[0] for _ in range(episodes)) / episodes


def cmd_qlearn(args):
    seed = int(args[0])
    rng, eval_rng = Rng(seed), Rng(seed + 1)
    bins = [1, 1, 6, 12]
    env = CartPole()
    q = [[0.0, 0.0] for _ in range(1 * 1 * 6 * 12)]
    greedy = lambda obs: argmax(q[discretize(obs, bins)])
    for ep in range(1, 2001):
        alpha = max(0.1, min(1.0, 1.0 - math.log10(ep / 25)))
        eps = max(0.05, min(1.0, 1.0 - math.log10(ep / 25)))
        obs = env.reset(rng)
        s = discretize(obs, bins)
        for t in range(env.limit):
            a = egreedy(rng, eps, q[s])
            obs, r, term = env.step(a)
            s2 = discretize(obs, bins)
            q_update(q, alpha, 0.99, s, a, r, s2, term)
            s = s2
            if term:
                break
        if ep % 50 == 0:
            score = evaluate(env, eval_rng, greedy, 10)
            print(f"episode {ep}: greedy average length {score:.1f}", file=sys.stderr)
            if score >= 475:
                print("solved")
                return
    print("not solved")


# ---------- stages 10-12: neural networks ----------

class MLP:
    def __init__(self, rng, sizes, act):
        self.params = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            bound = 1.0 / math.sqrt(fan_in)
            w = np.array([[rng.uniform(-bound, bound) for _ in range(fan_in)] for _ in range(fan_out)])
            b = np.array([rng.uniform(-bound, bound) for _ in range(fan_out)])
            self.params += [w, b]
        self.act = act

    def forward(self, x):
        self.cache = []
        h = np.atleast_2d(np.asarray(x, dtype=float))
        layers = len(self.params) // 2
        for i in range(layers):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            self.cache.append(h)
            z = h @ w.T + b
            if i < layers - 1:
                h = np.maximum(z, 0.0) if self.act == "relu" else np.tanh(z)
                self.cache.append(h)
            else:
                h = z
        return h

    def backward(self, dout):
        """Gradients of the last forward pass, given dLoss/dOutput (already averaged over the batch)."""
        grads = [None] * len(self.params)
        layers = len(self.params) // 2
        d = dout
        for i in reversed(range(layers)):
            w = self.params[2 * i]
            x = self.cache[2 * i]
            grads[2 * i] = d.T @ x
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = d @ w
                h = self.cache[2 * i - 1]
                d = d * (h > 0) if self.act == "relu" else d * (1.0 - h ** 2)
        return grads


class Optimizer:
    def __init__(self, params, kind, lr, clip=0.0):
        self.params, self.kind, self.lr, self.clip, self.t = params, kind, lr, clip, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        if self.clip > 0:
            norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
            coef = self.clip / (norm + 1e-6)
            if coef < 1:
                grads = [g * coef for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if self.kind == "sgd":
                p -= self.lr * g
            else:
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                mhat = m / (1 - 0.9 ** self.t)
                vhat = v / (1 - 0.999 ** self.t)
                p -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def sizes_arg(s):
    return [int(t) for t in s.split(",")]


def cmd_mlp(args):
    net = MLP(Rng(int(args[0])), sizes_arg(args[1]), args[2])
    for l in lines():
        print(fmts(net.forward(floats(l))[0]))


def loss_and_grad(d, kind):
    if kind == "mse":
        return d ** 2, 2 * d
    small = np.abs(d) < 1.0
    return np.where(small, 0.5 * d ** 2, np.abs(d) - 0.5), np.where(small, d, np.sign(d))


def cmd_train(args):
    net = MLP(Rng(int(args[0])), sizes_arg(args[1]), args[2])
    loss_kind, opt = args[3], Optimizer(net.params, args[4], float(args[5]), float(args[6]))
    batch = []
    for l in lines():
        batch = []
        for sample in l.split("|"):
            x, rest = sample.split(";")
            k, t = rest.split()
            batch.append((floats(x), int(k), float(t)))
        xs = np.array([b[0] for b in batch])
        ks = np.array([b[1] for b in batch])
        ts = np.array([b[2] for b in batch])
        out = net.forward(xs)
        d = out[np.arange(len(batch)), ks] - ts
        loss, g = loss_and_grad(d, loss_kind)
        dout = np.zeros_like(out)
        dout[np.arange(len(batch)), ks] = g / len(batch)
        opt.step(net.backward(dout))
        print(fmt(loss.mean()))
    for x, _, _ in batch:
        print(fmts(net.forward(x)[0]))


# ---------- stages 13-15: DQN ----------

class Replay:
    def __init__(self, capacity):
        self.capacity, self.items, self.pos = capacity, [], 0

    def add(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, rng, n):
        return [self.items[rng.randint(len(self.items))] for _ in range(n)]


def cmd_replay(args):
    buf, batch, rng = Replay(int(args[0])), int(args[1]), Rng(int(args[2]))
    for l in lines():
        op, _, rest = l.partition(" ")
        if op == "add":
            buf.add(rest.strip())
        elif op == "size":
            print(len(buf.items))
        else:
            print(" | ".join(buf.sample(rng, batch)))


def cmd_td(args):
    gamma, double = float(args[0]), len(args) > 1 and args[1] == "double"
    out = []
    for l in lines():
        parts = l.split("|")
        r, done = parts[0].split()
        if double:
            online, target = floats(parts[1]), floats(parts[2])
            nxt = target[argmax(online)]
        else:
            nxt = max(floats(parts[1]))
        out.append(float(r) + (0.0 if done == "1" else gamma * nxt))
    print(fmts(out))


def cmd_dqn(args):
    seed = int(args[0])
    rng, eval_rng = Rng(seed), Rng(seed + 1)
    env = CartPole()
    q = MLP(rng, [4, 64, 64, 2], "relu")
    target = MLP(rng, [4, 64, 64, 2], "relu")
    target.params = [p.copy() for p in q.params]
    opt = Optimizer(q.params, "adam", 1e-3, 10.0)
    buf = Replay(50000)
    batch, gamma, total = 64, 0.99, 60000
    greedy = lambda o: int(np.argmax(q.forward(o)[0]))
    obs = env.reset(rng)
    ep_len = 0
    for t in range(total):
        eps = linear_schedule(1.0, 0.05, 0.2, total, t)
        a = rng.randint(2) if rng.uniform() < eps else greedy(obs)
        nobs, r, term = env.step(a)
        ep_len += 1
        buf.add((obs, a, r, nobs, term))
        obs = nobs
        if term or ep_len >= env.limit:
            obs, ep_len = env.reset(rng), 0
        if t >= 1000:
            s = buf.sample(rng, batch)
            o = np.array([x[0] for x in s])
            acts = np.array([x[1] for x in s])
            rs = np.array([x[2] for x in s])
            o2 = np.array([x[3] for x in s])
            dones = np.array([x[4] for x in s], dtype=float)
            y = rs + gamma * (1 - dones) * target.forward(o2).max(axis=1)
            out = q.forward(o)
            idx = np.arange(batch)
            _, g = loss_and_grad(out[idx, acts] - y, "huber")
            dout = np.zeros_like(out)
            dout[idx, acts] = g / batch
            opt.step(q.backward(dout))
        if t % 500 == 0:
            target.params = [p.copy() for p in q.params]
        if t >= 1000 and t % 1000 == 0:
            score = evaluate(env, eval_rng, greedy, 10)
            print(f"step {t}: greedy average length {score:.1f}", file=sys.stderr)
            if score >= 195:
                print("solved")
                return
            obs, ep_len = env.reset(rng), 0
    print("not solved")


# ---------- stages 16-20: policy gradients ----------

def softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sample_categorical(rng, probs):
    u, c = rng.uniform(), 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def cmd_categorical(args):
    rng = Rng(int(args[0]))
    for l in lines():
        logits, a = l.split("|")
        p = softmax(floats(logits))
        logp = math.log(p[int(a)])
        ent = -float((p * np.log(p)).sum())
        print(fmts(p), "|", fmt(logp), fmt(ent), "|", sample_categorical(rng, p))


LOG2PI = math.log(2 * math.pi)


def cmd_gaussian(args):
    rng = Rng(int(args[0]))
    for l in lines():
        mu, logstd, a = (np.array(floats(p)) for p in l.split("|"))
        std = np.exp(logstd)
        logp = float((-0.5 * ((a - mu) / std) ** 2 - logstd - 0.5 * LOG2PI).sum())
        ent = float((0.5 + 0.5 * LOG2PI + logstd).sum())
        sample = [m + s * rng.normal() for m, s in zip(mu, std)]
        print(fmt(logp), fmt(ent), "|", fmts(sample))


def gae(rewards, values, dones, last_value, gamma, lam):
    adv = [0.0] * len(rewards)
    a, next_v = 0.0, last_value
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        a = delta + gamma * lam * nonterminal * a
        adv[t] = a
        next_v = values[t]
    return adv


def cmd_gae(args):
    gamma, lam = float(args[0]), float(args[1])
    ls = lines()
    steps = [l.split() for l in ls[:-1]]
    rewards = [float(s[0]) for s in steps]
    values = [float(s[1]) for s in steps]
    dones = [s[2] == "1" for s in steps]
    adv = gae(rewards, values, dones, float(ls[-1].split()[1]), gamma, lam)
    for a, v in zip(adv, values):
        print(fmt(a), fmt(a + v))


def ppo_terms(logp_new, logp_old, adv, clip, normalize):
    adv = np.asarray(adv, dtype=float)
    if normalize:
        adv = (adv - adv.mean()) / (adv.std(ddof=1) + 1e-8)
    log_ratio = np.asarray(logp_new) - np.asarray(logp_old)
    ratio = np.exp(log_ratio)
    unclipped, clipped = ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv
    loss = -float(np.minimum(unclipped, clipped).mean())
    clipfrac = float((np.abs(ratio - 1) > clip).mean())
    kl = float(((ratio - 1) - log_ratio).mean())
    return loss, clipfrac, kl


def cmd_ppo_loss(args):
    clip, normalize = float(args[0]), len(args) > 1 and args[1] == "norm"
    rows = [floats(l) for l in lines()]
    print(fmts(ppo_terms([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], clip, normalize)))


def ppo(seed, env, discrete, n_steps, epochs, minibatch, lr, gamma, lam, clip, budget, goal):
    rng, eval_rng = Rng(seed), Rng(seed + 1)
    obs_dim, act_dim = (4, 2) if discrete else (3, 1)
    actor = MLP(rng, [obs_dim, 64, 64, act_dim], "tanh")
    critic = MLP(rng, [obs_dim, 64, 64, 1], "tanh")
    actor.params[-2] *= 0.01  # tiny initial policy head: start close to uniform / zero-mean
    actor.params[-1] *= 0.0
    log_std = np.zeros(act_dim)
    params = actor.params + critic.params + ([] if discrete else [log_std])
    opt = Optimizer(params, "adam", lr, 0.5)

    def act(o):
        out = actor.forward(o)[0]
        if discrete:
            p = softmax(out)
            a = sample_categorical(rng, p)
            return a, math.log(p[a])
        a = out + np.exp(log_std) * np.array([rng.normal() for _ in range(act_dim)])
        return a, float((-0.5 * ((a - out) / np.exp(log_std)) ** 2 - log_std - 0.5 * LOG2PI).sum())

    def env_action(a):
        return a if discrete else float(a[0])

    def greedy(o):
        out = actor.forward(o)[0]
        return int(np.argmax(out)) if discrete else float(out[0])

    obs, ep_len, steps = env.reset(rng), 0, 0
    while steps < budget:
        buf_o, buf_a, buf_lp, buf_r, buf_v, buf_d = [], [], [], [], [], []
        for _ in range(n_steps):
            a, lp = act(obs)
            v = float(critic.forward(obs)[0, 0])
            nobs, r, term = env.step(env_action(a))
            ep_len += 1
            truncated = ep_len >= env.limit
            if truncated and not term:
                r += gamma * float(critic.forward(nobs)[0, 0])  # bootstrap through time-limit cut-offs
            buf_o.append(obs); buf_a.append(a); buf_lp.append(lp); buf_r.append(r); buf_v.append(v)
            buf_d.append(term or truncated)
            obs = nobs
            if term or truncated:
                obs, ep_len = env.reset(rng), 0
        steps += n_steps
        last_v = float(critic.forward(obs)[0, 0])
        adv = np.array(gae(buf_r, buf_v, buf_d, last_v, gamma, lam))
        ret = adv + np.array(buf_v)
        O, LP = np.array(buf_o), np.array(buf_lp)
        A = np.array(buf_a) if discrete else np.array(buf_a).reshape(-1, act_dim)
        for _ in range(epochs):
            order = list(range(n_steps))
            for i in range(n_steps - 1, 0, -1):  # Fisher-Yates with our own generator
                j = rng.randint(i + 1)
                order[i], order[j] = order[j], order[i]
            for start in range(0, n_steps, minibatch):
                idx = np.array(order[start:start + minibatch])
                n = len(idx)
                o, adv_b = O[idx], adv[idx]
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std(ddof=1) + 1e-8)
                out = actor.forward(o)
                if discrete:
                    p = softmax(out)
                    logp = np.log(p[np.arange(n), A[idx]])
                else:
                    std = np.exp(log_std)
                    z = (A[idx] - out) / std
                    logp = (-0.5 * z ** 2 - log_std - 0.5 * LOG2PI).sum(axis=1)
                ratio = np.exp(logp - LP[idx])
                use_unclipped = (ratio * adv_b) <= (np.clip(ratio, 1 - clip, 1 + clip) * adv_b)
                # d(-mean(min(...)))/dlogp: only the unclipped branch carries gradient
                dlogp = -(adv_b * ratio * use_unclipped) / n
                if discrete:
                    onehot = np.zeros_like(p)
                    onehot[np.arange(n), A[idx]] = 1
                    dout = dlogp[:, None] * (onehot - p)
                    g_logstd = []
                else:
                    dout = dlogp[:, None] * (z / std)
                    g_logstd = [(dlogp[:, None] * (z ** 2 - 1)).sum(axis=0)]
                g_actor = actor.backward(dout)
                vpred = critic.forward(o)[:, 0]
                dv = (2 * 0.5 * (vpred - ret[idx]) / n)[:, None]  # vf_coef 0.5 * mse
                g_critic = critic.backward(dv)
                opt.step(g_actor + g_critic + g_logstd)
        score = sum(run_episode(env, eval_rng, greedy)[1] for _ in range(10)) / 10
        print(f"step {steps}: greedy average return {score:.1f}", file=sys.stderr)
        if score >= goal:
            print("solved")
            return
        obs, ep_len = env.reset(rng), 0
    print("not solved")


def cmd_ppo_cartpole(args):
    ppo(int(args[0]), CartPole(), True, 1024, 10, 64, 1e-3, 0.99, 0.95, 0.2, 150000, 475)


def cmd_ppo_pendulum(args):
    ppo(int(args[0]), Pendulum(), False, 1024, 10, 64, 1e-3, 0.9, 0.95, 0.2, 300000, -250)


# ---------- stages 21-22: hyperparameter search ----------

def cmd_search(args):
    rng, n = Rng(int(args[0])), int(args[1])
    for i in range(n):
        lr = 10 ** rng.uniform(-5, -2)
        gamma = [0.9, 0.95, 0.98, 0.99, 0.999][rng.randint(5)]
        n_steps = [256, 512, 1024, 2048][rng.randint(4)]
        clip = rng.uniform(0.1, 0.3)
        print(f"trial {i}: learning_rate={lr:.3e} gamma={gamma} n_steps={n_steps} clip_range={clip:.4f}")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def cmd_prune(args):
    warmup = int(args[0])
    history, best = [], None
    for i, l in enumerate(lines()):
        curve = floats(l)
        reported = []
        for k, score in enumerate(curve, start=1):
            reported.append(score)
            peers = [h[k - 1] for h in history if len(h) >= k]
            if k > warmup and peers and score < median(peers):
                print(f"trial {i}: pruned at step {k}")
                break
        else:
            print(f"trial {i}: completed, final score {fmt(curve[-1])}")
            if best is None or curve[-1] > best[1]:
                best = (i, curve[-1])
        history.append(reported)
    print(f"best: trial {best[0]}" if best else "best: none")


COMMANDS = {
    "rng": cmd_rng, "cartpole": cmd_cartpole, "pendulum": cmd_pendulum, "random-agent": cmd_random_agent,
    "returns": cmd_returns, "schedule": cmd_schedule, "egreedy": cmd_egreedy, "discretize": cmd_discretize,
    "qupdate": cmd_qupdate, "qlearn": cmd_qlearn, "mlp": cmd_mlp, "train": cmd_train, "replay": cmd_replay,
    "td": cmd_td, "dqn": cmd_dqn, "categorical": cmd_categorical, "gaussian": cmd_gaussian, "gae": cmd_gae,
    "ppo-loss": cmd_ppo_loss, "ppo-cartpole": cmd_ppo_cartpole, "ppo-pendulum": cmd_ppo_pendulum,
    "search": cmd_search, "prune": cmd_prune,
}

if __name__ == "__main__":
    COMMANDS[sys.argv[1]](sys.argv[2:])
