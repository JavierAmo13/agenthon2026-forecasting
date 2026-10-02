import pandas as pd, pathlib

u = pathlib.Path('units')
seen = set()
for f in sorted(u.rglob('*.parquet')):
    if f.name in seen:
        continue
    seen.add(f.name)
    df = pd.read_parquet(f)
    print('=' * 20, f.parent.name, '/', f.name, df.shape)
    print(df.dtypes.to_dict())
    acol = next((c for c in ('asset', 'asset_id') if c in df.columns), None)
    if acol:
        print('assets:', sorted(df[acol].astype(str).unique()))
    print(df.head(3).to_string())
    print('date range:', df['date'].min(), '->', df['date'].max())
    print()