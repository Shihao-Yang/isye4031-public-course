import os
os.environ['MPLCONFIGDIR'] = '.mplcache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json

df = pd.read_csv('sales_territory.csv')
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
labels = ['Row'] + list(df.columns)
rows = [[str(i+1)] + [f'{v:,.2f}' for v in row] for i, row in enumerate(df.to_numpy())]
table = ax.table(cellText=rows, colLabels=labels, loc='center', cellLoc='right', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.65)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if r == 0:
        cell.set_facecolor('#dce6ef')
        cell.set_text_props(weight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f3f5f7')
ax.set_title('Sales-territory data: all 25 observations\nSales is the outcome; the other eight columns are predictors', fontsize=16, pad=18)
fig.tight_layout()
fig.savefig('sales_data_overview.png', dpi=180, bbox_inches='tight')
results = {'observations': len(df), 'columns': list(df.columns), 'missing_by_column': df.isna().sum().to_dict(), 'data': df.to_dict(orient='records')}
with open('sales_data_overview_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps({k:v for k,v in results.items() if k != 'data'}, indent=2))
