#!/usr/bin/env python3
"""
Streamlit-based interactive labeling UI for numeric datasets.

Run:
  streamlit run scripts/label_ui.py

Features:
 - Load a numeric CSV produced by convert_counter_csv_to_dataset.py
 - Preview rows, filter by sample_id, device metadata (if meta.json present)
 - Select rows and set labels interactively
 - Save labeled results to a file (in-place or new file)

This is intentionally simple — it's a light-weight annotator designed for quick
human-in-the-loop correction of numeric datasets.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json


def load_csv(path):
    df = pd.read_csv(path)
    return df


def save_csv(df, path):
    df.to_csv(path, index=False)


def main():
    st.title('AmdGpudrive — Labeler')

    st.sidebar.header('Dataset')
    csv_path = st.sidebar.text_input('Numeric CSV path', 'out_dataset.csv.numeric.csv')
    inplace = st.sidebar.checkbox('Edit in place', value=False)
    out_path = None
    if not inplace:
        out_path = st.sidebar.text_input('Output CSV path', 'out_dataset.labeled.csv')

    load_btn = st.sidebar.button('Load')

    if load_btn:
        if not os.path.exists(csv_path):
            st.sidebar.error('File not found: ' + csv_path)
            return
        df = load_csv(csv_path)

        st.sidebar.write('Rows:', len(df))

        # ensure label column exists
        if 'label' not in df.columns:
            df['label'] = 0

        # derived columns detection
        derived_cols = [c for c in df.columns if c not in ['sample_id', 'label'] and not c.startswith('counter_')]

        # active learning / uncertainty queue
        st.sidebar.header('Active learning')
        score_path = st.sidebar.text_input('Optional model scores CSV (sample_id,score)', '')
        build_queue = st.sidebar.button('Build uncertainty queue')
        queue_df = None
        if build_queue:
            queue_df = build_uncertainty_queue(df, score_path)
            st.sidebar.success(f'Queue built: {len(queue_df)} candidates')

        # show queue preview
        if queue_df is not None:
            st.sidebar.write(queue_df.head(20))

        # quick filtering
        sample_ids = sorted(map(str, df['sample_id'].astype(str).tolist()))
        sel = st.sidebar.multiselect('Select sample_id(s) to inspect', sample_ids, max_selections=50)

        # selection table
        if sel:
            sel_df = df[df['sample_id'].astype(str).isin(sel)].copy()
            st.subheader('Selected rows')
            st.dataframe(sel_df)
        else:
            st.subheader('Preview (first 50 rows)')
            st.dataframe(df.head(50))

        # label selection
        st.sidebar.header('Label')
        new_label = st.sidebar.number_input('New label value', min_value=0, max_value=999, value=1, step=1)
        apply_btn = st.sidebar.button('Apply label to selection')
        if apply_btn:
            if not sel:
                st.sidebar.error('No sample ids selected')
            else:
                idxs = df[df['sample_id'].astype(str).isin(sel)].index
                df.loc[idxs, 'label'] = new_label
                st.success(f'Applied label={new_label} to {len(idxs)} rows')

        # Save
        # Step through uncertain samples if queue is available
        if 'queue_idx' not in st.session_state:
            st.session_state.queue_idx = 0

        if queue_df is not None and len(queue_df) > 0:
            st.subheader('Uncertainty queue')
            st.write('Queue size:', len(queue_df))
            if st.button('Show next uncertain sample'):
                st.session_state.queue_idx = (st.session_state.queue_idx + 1) % len(queue_df)
            qi = st.session_state.queue_idx
            qsample = queue_df.iloc[qi]
            qsid = str(qsample['sample_id'])
            st.write(f'Queue position {qi+1}/{len(queue_df)} sample_id={qsid}')
            qdf = df[df['sample_id'].astype(str) == qsid]
            if not qdf.empty:
                st.dataframe(qdf)
                # display simple bar chart for numeric columns
                numeric_cols = [c for c in qdf.columns if c not in ['sample_id','label']]
                if numeric_cols:
                    vals = qdf.iloc[0][numeric_cols].replace({np.nan: None}).dropna()
                    if not vals.empty:
                        st.bar_chart(vals)

        save_btn = st.sidebar.button('Save')
        if save_btn:
            target = csv_path if inplace else (out_path or csv_path + '.labeled.csv')
            save_csv(df, target)
            st.success('Saved to ' + target)


def build_uncertainty_queue(df, score_csv=None, top_k=500):
    """
    Build a simple uncertainty queue. If score_csv provided, expects CSV with
    columns sample_id,score (score in [0,1], where 0.5 is most uncertain).
    Otherwise, fallback uses per-row average absolute z-score across numeric
    columns to select outliers (descending).
    Returns a dataframe with sample_id and an uncertainty score.
    """
    scores = None
    if score_csv and os.path.exists(score_csv):
        try:
            scores = pd.read_csv(score_csv)
            if 'sample_id' not in scores.columns or 'score' not in scores.columns:
                scores = None
        except Exception:
            scores = None

    # normalize sample_id to string
    df2 = df.copy()
    df2['sample_id'] = df2['sample_id'].astype(str)

    if scores is not None:
        scores['sample_id'] = scores['sample_id'].astype(str)
        # uncertainty = |score - 0.5| (lower = more uncertain)
        scores['uncertainty'] = scores['score'].apply(lambda s: abs(float(s) - 0.5))
        queue = scores.sort_values('uncertainty', ascending=True).head(top_k)[['sample_id', 'score', 'uncertainty']]
        return queue.reset_index(drop=True)

    # fallback: compute per-row mean absolute z-score across numeric columns
    num_cols = [c for c in df2.columns if c not in ['sample_id','label']]
    if not num_cols:
        return pd.DataFrame(columns=['sample_id', 'uncertainty'])
    col_means = df2[num_cols].mean(skipna=True)
    col_std = df2[num_cols].std(skipna=True).replace({0: 1.0})
    z = ((df2[num_cols] - col_means).abs() / col_std).mean(axis=1)
    qdf = pd.DataFrame({'sample_id': df2['sample_id'].values, 'uncertainty': z.values})
    # sort descending (large z = outlier/interesting first)
    qdf = qdf.sort_values('uncertainty', ascending=False).head(top_k).reset_index(drop=True)
    return qdf


if __name__ == '__main__':
    main()
