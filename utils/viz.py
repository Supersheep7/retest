from cfg import load_cfg
cfg = load_cfg()
import numpy as np
import pandas as pd
import einops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def get_direction(data, labels, pipeline):
    pipeline = clone(pipeline)
    pipeline.fit(data, labels)

    logreg = pipeline.named_steps["logreg"]
    return logreg.coef_[0] / np.linalg.norm(logreg.coef_[0])  # NO intercept


def get_direction_with_constraint(data, labels, pipeline, first_direction):
    # Orthogonalize in feature space
    proj = data @ first_direction
    data_orth = data - np.outer(proj, first_direction) / np.dot(first_direction, first_direction)

    pipeline = clone(pipeline)
    pipeline.fit(data_orth, labels)

    logreg = pipeline.named_steps["logreg"]
    return logreg.coef_[0] / np.linalg.norm(logreg.coef_[0])  # NO intercept

def act_for_viz(data, labels):

    assert data.shape[0] == labels.shape[0]

    if len(data.shape) > 2:
        data = einops.rearrange(data,
                                'n_batch batch_size d_model -> (n_batch batch_size) d_model')
    if len(labels.shape) > 1:
        labels = einops.rearrange(labels,
                                  'n_batch batch_size -> (n_batch batch_size)')

    data = np.array(data.cpu(), dtype=np.float32)
    labels = np.array(labels.cpu(), dtype=np.int32)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    pca_model = PCA(n_components=2)
    pca_projections = pca_model.fit_transform(data)

    pca_df = pd.DataFrame({
        'PCA1': pca_projections[:, 0],
        'PCA2': pca_projections[:, 1],
        'Label': ['False' if l == 0 else 'True' for l in labels]
    })

    model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(
                        max_iter=500,
                        n_jobs=-1,
                        solver="lbfgs",
                        multi_class="auto"
                    ))
                ])

    first_dir = get_direction(data, labels, model)
    first_proj = data @ first_dir  # no intercept
    second_dir = get_direction_with_constraint(data, labels, model, first_dir)
    second_proj = data @ second_dir  # no intercept

    probe_df = pd.DataFrame({
        'First direction': first_proj,
        'Second direction': second_proj,
        'Label': ['False' if l == 0 else 'True' for l in labels]
    })

    return pca_df, probe_df