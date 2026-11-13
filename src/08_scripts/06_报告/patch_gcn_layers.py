"""Add --gcn-layers to train.py so Fig.13 can sweep 1-5 propagation steps.

Default stays 3, so every existing checkpoint / command reproduces exactly.
"""
import ast

P = "train.py"
s = open(P, encoding="utf-8").read()

OLD_FLAG = '''    ap.add_argument("--gcn-mode", default="dynamic",
                    choices=["dynamic", "static", "none"],
                    help="dynamic semantic graph, fixed grid graph, or no GCN")
'''
NEW_FLAG = OLD_FLAG + '''    ap.add_argument("--gcn-layers", type=int, default=3,
                    help="number of MRConv layers in the refinement net "
                         "(Fig.13 propagation-step sweep)")
'''
assert OLD_FLAG in s, "gcn-mode block not found verbatim"
s = s.replace(OLD_FLAG, NEW_FLAG, 1)

OLD_CTOR = "k=args.k, residual=not args.no_residual).to(device)"
NEW_CTOR = ("k=args.k, gcn_layers=args.gcn_layers,\n"
            "                       residual=not args.no_residual).to(device)")
assert OLD_CTOR in s, "model ctor call not found"
s = s.replace(OLD_CTOR, NEW_CTOR, 1)

s = s.replace('f"afm={args.afm_mode} gcn={args.gcn_mode} k={args.k} "',
              'f"afm={args.afm_mode} gcn={args.gcn_mode} layers={args.gcn_layers} k={args.k} "')

ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("patched train.py")
