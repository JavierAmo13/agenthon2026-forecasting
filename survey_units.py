import json, pathlib, collections, tomllib

units = sorted(pathlib.Path('units').iterdir())
print('total units:', len(units))
fams = collections.Counter(); ttypes = collections.Counter(); horiz = collections.Counter()
nassets = collections.Counter(); panels_set = collections.Counter(); ndraws = collections.Counter()
tfreq = collections.Counter(); net = collections.Counter()
missing_spec = []
assets_all = collections.Counter()
for u in units:
    if not (u / 'forecast_spec.json').exists():
        missing_spec.append(u.name)
    card = tomllib.loads((u / 'card.toml').read_text(encoding='utf-8'))
    spec = json.loads((u / 'forecast_spec.json').read_text(encoding='utf-8')) if (u / 'forecast_spec.json').exists() else {}
    fam = spec.get('card_family') or card['metadata'].get('category')
    fams[fam] += 1
    t = card['targets']
    ttypes[t.get('target_type')] += 1
    horiz[tuple(t['horizons'])] += 1
    na = card['metadata'].get('n_assets', len(t['asset_ids']))
    nassets[na] += 1
    for a in t['asset_ids']:
        assets_all[a] += 1
    pp = tuple(sorted(p.name for p in u.glob('*.parquet')))
    panels_set[pp] += 1
    tfreq[t.get('target_frequency', card['metadata'].get('target_frequency'))] += 1
    net[card.get('environment', {}).get('network', '?')] += 1
print('families:', dict(fams))
print('target_types:', dict(ttypes))
print('target_freq:', dict(tfreq))
print('horizons:', dict(horiz))
print('n_assets:', dict(nassets))
print('panel files:', json.dumps({str(k): v for k, v in panels_set.items()}, indent=0))
print('network:', dict(net))
print('missing forecast_spec.json:', missing_spec)
print('unique assets:', len(assets_all), dict(assets_all.most_common(40)))