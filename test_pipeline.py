import sys
sys.path.insert(0, '.')
from custom_model.data.loader import load_unit
from custom_model.data.features import build_features
from custom_model.data.targets import build_all_targets
from custom_model.data.dataset import build_dataset

b = load_unit('units/t2-F3-boj-ust-channel-2023')
print('card:', b.card_id, '| family:', b.card_family, '| asof:', b.asof)
print('assets:', b.target_assets, '| horizons:', b.horizons,
      '| type:', b.target_type, '| freq:', b.target_frequency)
print('panels:', {k: v.shape for k, v in b.panels.items()})
print('texts:', len(b.texts))

X = build_features(b.panels, b.asof)
print('features:', X.shape)
print(X.columns[:12].tolist())

T = build_all_targets(b)
print('targets:', T.shape, T.columns.tolist())

ds = build_dataset(X, T, 'UST_10Y', 21)
print('dataset UST_10Y h21: X', ds.X.shape, 'y', ds.y.shape,
      '| NaN in X:', int(ds.X.isna().sum().sum()),
      '| X_pred:', ds.X_pred.shape)