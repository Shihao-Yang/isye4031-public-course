import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def weights(c1, c2, n=26):
    w = np.zeros(n)
    w[0] = 1.0
    for j in range(1, n):
        w[j] = c1*w[j-1] + (c2*w[j-2] if j > 1 else 0.0)
    return w


def check(c1, c2):
    roots = np.roots([-c2, -c1, 1.0])
    tests = [c2 > -1, c1+c2 < 1, c2-c1 < 1]
    return {'c1': c1, 'c2': c2,
            'roots': [{'real': float(r.real), 'imag': float(r.imag), 'modulus': float(abs(r))} for r in roots],
            'inequalities': [bool(t) for t in tests], 'passes': bool(all(tests)),
            'weights': weights(c1, c2).tolist()}


def main():
    cases = [('A', 0.6, 0.2), ('B', 1.2, -0.5), ('C', 0.8, 0.4), ('D', 0.5, -1.2)]
    results = {name: check(c1, c2) for name, c1, c2 in cases}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for ax, (name, c1, c2) in zip(axes.flat, cases):
        result = results[name]
        w = np.array(result['weights'])
        ax.axhline(0, color='0.6', linewidth=0.8)
        ax.stem(np.arange(len(w)), w, basefmt=' ')
        ax.set_title(f'{name}: c1={c1}, c2={c2} | ' + ('stable' if result['passes'] else 'unstable'))
        ax.set_xlabel('Lag j')
        ax.set_ylabel('Expansion weight w_j')
        ax.grid(alpha=0.2)
    fig.suptitle('Constructed second-order examples: inverse of 1 - c1 B - c2 B²\nAR causal weights or MA innovation-recovery weights', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig('turn03_weights.png', dpi=170)
    plt.close(fig)
    Path('turn03_weights.json').write_text(json.dumps({'description': 'Fixed illustrative parameters, no data fitted. Weights satisfy w0=1, w1=c1, wj=c1*w(j-1)+c2*w(j-2).', 'cases': results}, indent=2))
    for name, result in results.items():
        print(name, 'root moduli:', [round(r['modulus'], 4) for r in result['roots']], 'tests:', result['inequalities'])


if __name__ == '__main__':
    main()
