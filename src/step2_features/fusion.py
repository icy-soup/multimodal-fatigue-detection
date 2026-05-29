"""多模态特征聚合器

整合 EEG(74) + ECG(12) + EDA(6) = 92维
"""

import numpy as np
from . import eeg, ecg, eda


def extract_session(session_data):
    n = len(session_data['labels'])

    eeg_f = eeg.extract_batch(session_data['eeg'])
    ecg_f = ecg.extract_batch(session_data['ecg'])
    eda_f = eda.extract_batch(session_data['eda'], session_data.get('scl'),
                               session_data.get('scr'), session_data.get('scr_peaks'))

    dims = (eeg_f.shape[1], ecg_f.shape[1], eda_f.shape[1])
    print(f"  EEG:{eeg_f.shape}  ECG:{ecg_f.shape}  EDA:{eda_f.shape}  Total:{eeg_f.shape[1]+ecg_f.shape[1]+eda_f.shape[1]}d")

    return np.concatenate([eeg_f, ecg_f, eda_f], axis=1), session_data['labels'], dims


def extract_dataset(all_data):
    X_list, y_list, dims = [], [], None
    for d in all_data:
        X, y, dims = extract_session(d)
        X_list.append(X), y_list.append(y)

    from itertools import chain
    subj_ids = list(chain.from_iterable([d['subject']] * len(d['labels']) for d in all_data))

    return {'X': np.vstack(X_list), 'y': np.concatenate(y_list),
            'modality_dims': dims, 'subject_ids': subj_ids}
