"""
Preflight check for the full-system evaluation: every stem needs a wandb_id that resolves to
exactly one stem restoration checkpoint under <ckpt_root>/*/<stem>/<wandb_id>/last.ckpt --
otherwise ApolloStemwiseRestorationWrapper would silently fall back to the shared pretrained
Apollo model for that stem.

Usage: python scripts/evaluation/check_stem_restoration_ckpts.py <ckpt_root> "<ckpt_ids dict>"
"""
import ast
import glob
import os
import sys

STEMS = ['bass', 'drums', 'vocals', 'guitars', 'keyboards', 'synthesizers', 'percussions', 'orchestral']


def main(root, ckpt_ids):
    ids = ast.literal_eval(ckpt_ids)
    ok = True
    for stem in STEMS:
        wandb_id = ids.get(stem)
        if not wandb_id or wandb_id == 'TODO':
            ok = False
            print(f'[ckpt] ERROR {stem}: no wandb_id set')
            continue
        matches = glob.glob(os.path.join(root, '*', stem, wandb_id, 'last.ckpt'))
        if len(matches) == 1:
            print(f'[ckpt] {stem}: {matches[0]}')
        else:
            ok = False
            print(f'[ckpt] ERROR {stem}: wandb_id {wandb_id!r} matched {len(matches)} checkpoints under {root}/*/{stem}: {matches}')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
