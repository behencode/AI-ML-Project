"""Exploratory Data Analysis utilities for the RACE project.

Generates basic EDA figures and a small summary CSV for inclusion in the report.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def basic_eda(raw_dir: str = os.path.join('data', 'raw'), out_dir: str = os.path.join('reports', 'figures')) -> dict:
    ensure_dir(out_dir)
    paths = {split: os.path.join(raw_dir, f'{split}.csv') for split in ['train', 'val', 'test']}
    df = pd.read_csv(paths['train'])
    summary = {}

    # Missing values
    missing = df.isna().sum().to_dict()
    summary['missing'] = missing

    # Length distributions
    df['article_len'] = df['article'].fillna('').str.split().apply(len)
    df['question_len'] = df['question'].fillna('').str.split().apply(len)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['article_len'].clip(0, 1000), bins=50)
    ax.set_title('Article length (words)')
    ax.set_xlabel('Words')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'article_length_hist.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['question_len'].clip(0, 100), bins=40)
    ax.set_title('Question length (words)')
    ax.set_xlabel('Words')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'question_length_hist.png'))
    plt.close(fig)

    # Answer distribution
    ans_counts = df['answer'].fillna('NA').value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ans_counts.plot(kind='bar', ax=ax)
    ax.set_title('Answer label distribution')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'answer_label_distribution.png'))
    plt.close(fig)

    # Save a small summary CSV
    summary_df = pd.DataFrame({'metric': ['n_rows', 'mean_article_len', 'mean_question_len'], 'value': [len(df), float(df['article_len'].mean()), float(df['question_len'].mean())]})
    summary_df.to_csv(os.path.join(out_dir, 'eda_summary.csv'), index=False)

    summary.update({'n_rows': len(df), 'mean_article_len': float(df['article_len'].mean()), 'mean_question_len': float(df['question_len'].mean())})
    return summary


def main() -> None:
    s = basic_eda()
    print('EDA summary:')
    print(s)


if __name__ == '__main__':
    main()
