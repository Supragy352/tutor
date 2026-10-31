import math
import sys


def fmt(v):
    s = f"{v:.4f}"
    return "0.0000" if s == "-0.0000" else s


def parse_kv(line, defaults):
    out = dict(defaults)
    for tok in line.split():
        k, v = tok.split("=", 1)
        out[k] = float(v)
    return out


CTRL_DEFAULTS = {"kp": 0.0, "ki": 0.0, "kd": 0.0, "dt": 1.0, "min": -math.inf, "max": math.inf}
PLANT_DEFAULTS = {"gain": 1.0, "tau": 1.0, "y0": 0.0}


class PID:
    def __init__(self, c):
        self.c = c
        self.integral = 0.0
        self.prev_meas = None

    def step(self, sp, meas):
        c = self.c
        e = sp - meas
        d = 0.0
        if self.prev_meas is not None:
            d = -c["kd"] * (meas - self.prev_meas) / c["dt"]
        self.prev_meas = meas
        pd = c["kp"] * e + d
        tentative = self.integral + e * c["dt"]
        u = pd + c["ki"] * tentative
        # conditional integration: don't integrate further into saturation
        if not ((u > c["max"] and e > 0) or (u < c["min"] and e < 0)):
            self.integral = tentative
        u = pd + c["ki"] * self.integral
        return min(max(u, c["min"]), c["max"]), e


def simulate(c, p, segments, emit=None):
    pid = PID(c)
    y = p["y0"]
    iae = 0.0
    for sp, steps in segments:
        for _ in range(steps):
            u, e = pid.step(sp, y)
            iae += abs(e) * c["dt"]
            y += c["dt"] * (p["gain"] * u - y) / p["tau"]
            if emit:
                emit(f"{fmt(y)} {fmt(u)}")
    return iae


def segments_of(lines):
    return [(float(a), int(b)) for a, b in (l.split() for l in lines)]


def main():
    lines = [l for l in sys.stdin.read().splitlines() if l.strip()]
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    c = parse_kv(lines[0], CTRL_DEFAULTS)
    if mode == "--sim":
        simulate(c, parse_kv(lines[1], PLANT_DEFAULTS), segments_of(lines[2:]), print)
    elif mode == "--tune":
        p = parse_kv(lines[1], PLANT_DEFAULTS)
        segs = segments_of(lines[2:3])
        best = None
        for i, line in enumerate(lines[3:], 1):
            kp, ki, kd = map(float, line.split())
            iae = simulate({**c, "kp": kp, "ki": ki, "kd": kd}, p, segs)
            print(f"{i} {fmt(iae)}")
            if best is None or iae < best[1]:
                best = (i, iae)
        print(f"best {best[0]}")
    else:
        pid = PID(c)
        for line in lines[1:]:
            sp, m = map(float, line.split())
            print(fmt(pid.step(sp, m)[0]))


if __name__ == "__main__":
    main()
