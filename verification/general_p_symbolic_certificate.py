#!/usr/bin/env python3

"""Exact symbolic general-p Hall certificate for the Bernoulli REG/IREG proof.

Uses formal Einstein indices, exact rational arithmetic, and symbolic p.
No coefficient is inferred by fitting across dimensions.
"""
from dataclasses import dataclass
from fractions import Fraction
from collections import defaultdict, Counter
import itertools
import sympy as sp

# Small scalar-indexed expression algebra.
# A scalar is represented as a sparse sum of monomials.  A monomial consists of
# a rational coefficient and tensor factors (name, ordered index tuple).
# Repeated integer labels are Einstein-summed labels.  We never attach a finite
# range to them; the only dimensional identity is delta[i,i] = p.

SYMMETRIC = {'Sigma', 'K', 'M', 'P', 'T3', 'T4', 'S', 'Z', 'R', 'U'}

class Idx:
    __slots__ = ('n',)
    def __init__(self, n): self.n = int(n)
    def __repr__(self): return f'i{self.n}'

@dataclass(frozen=True)
class Factor:
    name: str
    inds: tuple[int, ...]

@dataclass
class Term:
    coeff: Fraction
    factors: tuple[Factor, ...]

class Expr:
    def __init__(self, terms=None):
        self.terms = defaultdict(Fraction)
        if terms:
            for k,v in terms.items():
                if v: self.terms[k] += Fraction(v)
        self.clean()
    @staticmethod
    def zero(): return Expr()
    @staticmethod
    def one(): return Expr({(): Fraction(1)})
    @staticmethod
    def scalar(c):
        c=Fraction(c)
        return Expr({():c}) if c else Expr.zero()
    @staticmethod
    def atom(name, *inds):
        return Expr({(Factor(name, tuple(_idval(i) for i in inds)),):Fraction(1)})
    def copy(self): return Expr(dict(self.terms))
    def clean(self):
        self.terms = defaultdict(Fraction, {k:v for k,v in self.terms.items() if v})
        return self
    def __add__(self, other):
        other=to_expr(other); out=self.copy()
        for k,v in other.terms.items(): out.terms[k]+=v
        return out.clean()
    __radd__=__add__
    def __neg__(self): return self.scale(-1)
    def __sub__(self, other): return self + (-to_expr(other))
    def __rsub__(self, other): return to_expr(other)-self
    def __mul__(self, other):
        other=to_expr(other)
        if not self.terms or not other.terms: return Expr.zero()
        out=defaultdict(Fraction)
        for k1,c1 in self.terms.items():
            for k2,c2 in other.terms.items():
                fac=k1+k2
                fac=tuple(sorted(fac,key=lambda f:(f.name,f.inds)))
                out[fac]+=c1*c2
        return Expr(out).simplify_local()
    __rmul__=__mul__
    def scale(self,c):
        c=Fraction(c)
        return Expr({k:c*v for k,v in self.terms.items()})
    def __bool__(self): return bool(self.terms)
    def simplify_local(self):
        # Canonicalize slots of symmetric tensors; don't rename indices here.
        out=defaultdict(Fraction)
        for facs,c in self.terms.items():
            nf=[]
            for f in facs:
                inds=tuple(sorted(f.inds)) if f.name in SYMMETRIC else f.inds
                nf.append(Factor(f.name,inds))
            out[tuple(sorted(nf,key=lambda f:(f.name,f.inds)))]+=c
        self.terms=out
        return self.clean()
    def num_terms(self): return len(self.terms)
    def __repr__(self): return f'Expr({len(self.terms)} terms)'

def _idval(x): return x.n if isinstance(x,Idx) else int(x)

def to_expr(x):
    if isinstance(x,Expr): return x
    return Expr.scalar(x)

class IndexPool:
    def __init__(self): self.next=0
    def fresh(self):
        i=self.next; self.next+=1; return i
    def many(self,n): return [self.fresh() for _ in range(n)]

# Rewriting: K Sigma -> delta; delta substitution; closed delta trace -> p.
# These are ordinary indexed identities, not invariant/topology classification.

def factor_counts(facs):
    c=Counter()
    for f in facs:
        c.update(f.inds)
    return c

def substitute_index(facs, old, new):
    return tuple(Factor(f.name, tuple(new if x==old else x for x in f.inds)) for f in facs)

def rewrite_delta_term(facs, coeff):
    facs=list(facs)
    changed=True
    while changed:
        changed=False
        # delta contraction / trace
        for pos,f in enumerate(facs):
            if f.name!='Delta': continue
            a,b=f.inds
            if a==b:
                facs.pop(pos); coeff*=1  # p handled below by explicit marker
                # represent dimension loop as Dim factor
                facs.append(Factor('Dim',()))
                changed=True; break
            counts=factor_counts(facs)
            # If either side is otherwise absent, delta is a free identity factor;
            # final Hall scalars should not leave one.  If both occur elsewhere,
            # substitute b -> a when possible.
            if counts[a]==1 and counts[b]>=2:
                facs.pop(pos); facs=list(substitute_index(facs,b,a)); changed=True; break
            if counts[b]==1 and counts[a]>=2:
                facs.pop(pos); facs=list(substitute_index(facs,a,b)); changed=True; break
        if changed: continue
        counts=factor_counts(facs)
        # K-Sigma adjacent contraction with a summed shared index.
        found=False
        for i,f in enumerate(facs):
            if f.name!='K': continue
            for j,g in enumerate(facs):
                if j==i or g.name!='Sigma': continue
                common=set(f.inds)&set(g.inds)
                for m in common:
                    if counts[m] != 2: continue
                    a=f.inds[0] if f.inds[1]==m else f.inds[1]
                    b=g.inds[0] if g.inds[1]==m else g.inds[1]
                    # replace K(a,m) Sigma(m,b) by Delta(a,b)
                    rem=[h for k,h in enumerate(facs) if k not in (i,j)]
                    rem.append(Factor('Delta',(a,b)))
                    facs=rem; changed=True; found=True; break
                if found: break
            if found: break
    return tuple(sorted(facs,key=lambda f:(f.name,f.inds))), coeff

def reduce_expr(e:Expr):
    out=defaultdict(Fraction)
    for facs,c in e.terms.items():
        nf,nc=rewrite_delta_term(facs,c)
        # Replace Dim by scalar p only in final pretty algebra.  Keep symbol P_DIM
        # as an atomic scalar coefficient via SymPy for exact p-polynomials.
        n_dim=sum(1 for f in nf if f.name=='Dim')
        nf=tuple(f for f in nf if f.name!='Dim')
        out[nf]+=nc*Fraction(1) * sp.Integer(1) if n_dim==0 else nc
    return Expr(out)

# We need symbolic p coefficients, so a second lightweight representation is
# used for terms after dimension traces.  Coefficients are SymPy polynomials.
class PExpr:
    def __init__(self, terms=None):
        self.terms=defaultdict(lambda:sp.Integer(0))
        if terms:
            for k,v in terms.items(): self.terms[k]+=sp.sympify(v)
        self.clean()
    @staticmethod
    def zero(): return PExpr()
    @staticmethod
    def one(): return PExpr({():1})
    @staticmethod
    def scalar(c): return PExpr({():sp.sympify(c)}) if c else PExpr.zero()
    @staticmethod
    def atom(name,*inds):
        fac=Factor(name,tuple(_idval(i) for i in inds))
        return PExpr({(fac,):1})
    def clean(self): self.terms=defaultdict(lambda:sp.Integer(0),{k:sp.expand(v) for k,v in self.terms.items() if v!=0}); return self
    def __add__(self,o):
        o=to_pexpr(o); d=dict(self.terms)
        for k,v in o.terms.items(): d[k]=d.get(k,0)+v
        return PExpr(d)
    __radd__=__add__
    def __neg__(self): return self.scale(-1)
    def __sub__(self,o): return self+(-to_pexpr(o))
    def __rsub__(self,o): return to_pexpr(o)-self
    def __mul__(self,o):
        o=to_pexpr(o)
        d=defaultdict(lambda:sp.Integer(0))
        for k1,c1 in self.terms.items():
            for k2,c2 in o.terms.items():
                fs=tuple(sorted(k1+k2,key=lambda f:(f.name,f.inds)))
                d[fs]+=c1*c2
        return PExpr(d).local_sym()
    __rmul__=__mul__
    def scale(self,c): return PExpr({k:sp.expand(sp.sympify(c)*v) for k,v in self.terms.items()})
    def local_sym(self):
        d=defaultdict(lambda:sp.Integer(0))
        for fs,c in self.terms.items():
            nfs=[]
            for f in fs:
                inds=tuple(sorted(f.inds)) if f.name in SYMMETRIC else f.inds
                nfs.append(Factor(f.name,inds))
            d[tuple(sorted(nfs,key=lambda f:(f.name,f.inds)))]+=c
        return PExpr(d)
    def num_terms(self): return len(self.terms)

def to_pexpr(x):
    if isinstance(x,PExpr): return x
    return PExpr.scalar(x)

def pexpr_rewrite_term(facs, coeff):
    facs=list(facs); coeff=sp.sympify(coeff)
    changed=True
    while changed:
        changed=False
        counts=factor_counts(facs)
        for pos,f in enumerate(facs):
            if f.name!='Delta': continue
            a,b=f.inds
            if a==b:
                facs.pop(pos); coeff*=sp.Symbol('p'); changed=True; break
            if counts[a]==1 and counts[b]>=2:
                facs.pop(pos); facs=list(substitute_index(facs,b,a)); changed=True; break
            if counts[b]==1 and counts[a]>=2:
                facs.pop(pos); facs=list(substitute_index(facs,a,b)); changed=True; break
        if changed: continue
        counts=factor_counts(facs); found=False
        for i,f in enumerate(facs):
            if f.name!='K': continue
            for j,g in enumerate(facs):
                if j==i or g.name!='Sigma': continue
                for m in set(f.inds)&set(g.inds):
                    if counts[m]!=2: continue
                    a=f.inds[0] if f.inds[1]==m else f.inds[1]
                    b=g.inds[0] if g.inds[1]==m else g.inds[1]
                    facs=[h for k,h in enumerate(facs) if k not in (i,j)]
                    facs.append(Factor('Delta',(a,b))); found=True; changed=True; break
                if found: break
            if found: break
    return tuple(sorted(facs,key=lambda f:(f.name,f.inds))),sp.expand(coeff)

def reduce_pexpr(e:PExpr):
    d=defaultdict(lambda:sp.Integer(0))
    for fs,c in e.terms.items():
        nfs,nc=pexpr_rewrite_term(fs,c); d[nfs]+=nc
    return PExpr(d)

# Canonicalization of dummy labels.  We use a deterministic backtracking
# canonicalizer on the finite set of labels occurring in a monomial.  Tensor
# factors have names and ordered slots; symmetric slots were sorted first.
# For the sizes produced here, WL hash is used as a quick certificate and the
# exact isomorphism fallback is tiny.

def canonical_factor_key(f): return (f.name, f.inds)

def canon_renaming(factors):
    # Refinement by factor-incidence signatures. Labels are dummy; choose a
    # stable canonical renaming by repeated colors. In all expressions generated
    # here, each index is Einstein-summed (appears exactly twice) after all
    # contractions, which makes this refinement particularly strong.
    labels=sorted(set(i for f in factors for i in f.inds))
    if not labels: return tuple(factors)
    occ={i:[] for i in labels}
    for fpos,f in enumerate(factors):
        for slot,i in enumerate(f.inds): occ[i].append((fpos,f.name,slot))
    # Initial signature ignores actual label values.
    colors={i:tuple(sorted((name,slot) for _,name,slot in occ[i])) for i in labels}
    # Refine through neighboring factor labels.
    for _ in range(8):
        new={}
        for i in labels:
            neigh=[]
            for fpos,name,slot in occ[i]:
                f=factors[fpos]
                neigh.append((name,slot,tuple(colors[j] for j in f.inds)))
            new[i]=tuple(sorted(neigh,key=repr))
        if new==colors: break
        colors=new
    # Sort indices by color, then by a stable occurrence description.
    order=sorted(labels,key=lambda i:(repr(colors[i]),repr(occ[i])))
    ren={old:new for new,old in enumerate(order)}
    return tuple(sorted((Factor(f.name,tuple(ren[i] for i in f.inds)) for f in factors),key=lambda f:(f.name,f.inds)))

def canonicalize(e:PExpr):
    d=defaultdict(lambda:sp.Integer(0))
    for fs,c in e.terms.items():
        fs2=canon_renaming(fs)
        d[fs2]+=c
    return PExpr(d)

def exact_zero(e:PExpr):
    e=canonicalize(reduce_pexpr(e))
    return not e.terms

# Indexed block-matrix algebra for R, dA, db and the exact derivative formulas.
# Block categories use S, W, X(i), WX(i).  A coordinate direction itself is a
# (type, tuple-of-formal-indices).  Matrix multiplication introduces fresh
# summed feature labels for X/WX intermediate blocks.

class BlockMat:
    def __init__(self, rows, cols, entries=None):
        self.rows=rows; self.cols=cols; self.entries=entries or {}
    def get(self,r,c): return self.entries.get((r,c), PExpr.zero())

def cat_scalar(s): return ('s',s)
def cat_X(i): return ('x',i)
def cat_WX(i): return ('wx',i)

REG_R = {}
REG_R[(cat_scalar('0'),cat_scalar('0'))]=PExpr.scalar(2)
REG_R[(cat_scalar('0'),cat_scalar('w'))]=PExpr.scalar(-2)
REG_R[(cat_scalar('w'),cat_scalar('0'))]=PExpr.scalar(-2)
REG_R[(cat_scalar('w'),cat_scalar('w'))]=PExpr.scalar(4)
def make_R_reg():
    R=BlockMat(['0','w','x'],['0','w','x'],dict(REG_R))
    # vector-vector entry is handled as a symbolic function, not stored.
    return R

def make_R_ireg():
    ent={
      (cat_scalar('0'),cat_scalar('0')):PExpr.scalar(2),
      (cat_scalar('0'),cat_scalar('w')):PExpr.scalar(-2),
      (cat_scalar('w'),cat_scalar('0')):PExpr.scalar(-2),
      (cat_scalar('w'),cat_scalar('w')):PExpr.scalar(4),
    }
    # x-x, x-wx, wx-x, wx-wx handled by generic get.
    return BlockMat(['0','w','x','wx'],['0','w','x','wx'],ent)

def Rget(ireg,row,col):
    rt,ri=row; ct,ci=col
    if rt=='s' and ct=='s':
        vals={('0','0'):2,('0','w'):-2,('w','0'):-2,('w','w'):4}
        return PExpr.scalar(vals.get((ri,ci),0))
    if rt=='x' and ct=='x': return PExpr.atom('K',ri,ci).scale(2 if ireg else 1)
    if ireg and rt=='x' and ct=='wx': return PExpr.atom('K',ri,ci).scale(-2)
    if ireg and rt=='wx' and ct=='x': return PExpr.atom('K',ri,ci).scale(-2)
    if ireg and rt=='wx' and ct=='wx': return PExpr.atom('K',ri,ci).scale(4)
    return PExpr.zero()

def categories(ireg, free_label=None):
    i=free_label if free_label is not None else '_'
    cats=[cat_scalar('0'),cat_scalar('w')]
    # For componentwise vectors, a concrete feature label is supplied when a
    # row/column is requested.  The generic intermediate sum introduces fresh i.
    return cats

def matvec_R(ireg, vfun, outcat, pool):
    # R*v, summing over intermediate regressor coordinate category.
    total=PExpr.zero()
    for mid_type in ['0','w','x'] + (['wx'] if ireg else []):
        if mid_type in ('0','w'):
            mids=[cat_scalar(mid_type)]
        else:
            q=pool.fresh(); mids=[cat_X(q) if mid_type=='x' else cat_WX(q)]
        for mid in mids:
            e=Rget(ireg,outcat,mid)
            if e: total += e*vfun(mid)
    return reduce_pexpr(total)

def matmul(ireg, A, B, row, col, pool):
    total=PExpr.zero()
    for mid_type in ['0','w','x'] + (['wx'] if ireg else []):
        if mid_type in ('0','w'): mid=cat_scalar(mid_type)
        else:
            q=pool.fresh(); mid=cat_X(q) if mid_type=='x' else cat_WX(q)
        ea=A(row,mid,pool); eb=B(mid,col,pool)
        if ea and eb: total += ea*eb
    return reduce_pexpr(total)

def direction_dA(ireg, dtyp, labels):
    # sparse entry function row,col -> PExpr.  labels are the feature labels
    # carried by this coordinate direction.
    def get(row,col,pool=None):
        rt,ri=row; ct,ci=col
        z=PExpr.zero()
        if dtyp=='X':
            i=labels[0]
            if (rt,ct) in [('s','x') ,('x','s')] and (ri==i if rt=='x' else ci==i): return PExpr.one()
            return z
        if dtyp=='XX':
            i,j=labels
            if rt=='x' and ct=='x' and ((ri==i and ci==j) or (ri==j and ci==i)): return PExpr.one()
            return z
        if dtyp=='W':
            if (rt,ri,ct,ci)==('s','0','s','0'): return PExpr.one()
            if (rt,ri,ct,ci)==('s','0','s','w'): return PExpr.one()
            if (rt,ri,ct,ci)==('s','w','s','0'): return PExpr.one()
            if (rt,ri,ct,ci)==('s','w','s','w'): return PExpr.one()
            return z
        if dtyp=='WX':
            i=labels[0]
            # REG: X_i-W block. IREG: same plus W-WX_i and intercept-WX_i.
            if (rt,ct)==('x','s') and ri==i and ci=='w': return PExpr.one()
            if (rt,ct)==('s','x') and ri=='w' and ci==i: return PExpr.one()
            if ireg:
                if (rt,ct)==('s','wx') and ri=='0' and ci==i: return PExpr.one()
                if (rt,ct)==('wx','s') and ri==i and ci=='0': return PExpr.one()
                if (rt,ct)==('s','wx') and ri=='w' and ci==i: return PExpr.one()
                if (rt,ct)==('wx','s') and ri==i and ci=='w': return PExpr.one()
            return z
        if dtyp=='X2W':
            i,j=labels
            if not ireg: return z
            if rt=='x' and ct=='wx':
                if (ri==i and ci==j) or (ri==j and ci==i): return PExpr.one()
            if rt=='wx' and ct=='x':
                if (ri==i and ci==j) or (ri==j and ci==i): return PExpr.one()
            if rt=='wx' and ct=='wx' and ((ri==i and ci==j) or (ri==j and ci==i)): return PExpr.one()
            return z
        return z
    return get

def direction_db(ireg,dtyp,labels):
    def get(cat,pool=None):
        if dtyp=='Y' and cat==cat_scalar('0'): return PExpr.one()
        if dtyp=='XY' and cat[0]=='x' and cat[1]==labels[0]: return PExpr.one()
        if dtyp=='WY' and cat==cat_scalar('w'): return PExpr.one()
        if ireg and dtyp=='XWY' and cat[0]=='wx' and cat[1]==labels[0]: return PExpr.one()
        return PExpr.zero()
    return get

def theta0(ireg):
    # Canonical equivalent representative Y = W gamma'X + eps.
    # REG theta_X = gamma/2; IREG theta_WX = gamma.
    def get(cat):
        if cat[0]=='x': return PExpr.atom('gamma',cat[1]).scale(Fraction(1,2))
        if ireg and cat[0]=='wx': return PExpr.atom('gamma',cat[1])
        return PExpr.zero()
    return get

def vec_linear_form(Rmultfun, e):
    return Rmultfun(e)

def dtheta(ireg,dtyp,labels,pool):
    Rpool=pool
    dA=direction_dA(ireg,dtyp,labels); db=direction_db(ireg,dtyp,labels); th=theta0(ireg)
    def rhs(cat):
        total=db(cat)
        # subtract dA * theta
        def v(mid): return th(mid)
        da=matvec_generic(ireg,dA,v,cat,pool)
        return total-da
    def out(cat):
        def v(mid): return rhs(mid)
        return matvec_R(ireg,v,cat,pool)
    return out

def matvec_generic(ireg,A,vfun,row,pool):
    total=PExpr.zero()
    for mt in ['0','w','x'] + (['wx'] if ireg else []):
        mid=cat_scalar(mt) if mt in ('0','w') else (cat_X(pool.fresh()) if mt=='x' else cat_WX(pool.fresh()))
        e=A(row,mid,pool); vv=vfun(mid)
        if e and vv: total += e*vv
    return reduce_pexpr(total)

def base_a(ireg,cat):
    # R e_W
    return {'0':PExpr.scalar(-2),'w':PExpr.scalar(4)}.get(cat[1] if cat[0]=='s' else '',PExpr.zero())

def a_direction(ireg,dtyp,labels,pool):
    # a = R e_W; extended IREG endpoint correction for X directions only.
    dA=direction_dA(ireg,dtyp,labels); db=direction_db(ireg,dtyp,labels)
    dt=dtheta(ireg,dtyp,labels,pool)
    # R dterm, then e_W^T times it = component W of R dterm.
    base=matvec_generic(ireg, lambda r,c,p: Rget(ireg,r,c), lambda c: db(c)-matvec_generic(ireg,dA,lambda m:theta0(ireg)(m),c,pool), cat_scalar('w'), pool)
    # Above is e_W^T R (db-dA theta), but R symmetric; okay.
    if ireg and dtyp=='X':
        i=labels[0]
        # dc_i * (e_aux_i^T theta0), where theta0 WX_i=gamma_i.
        base += PExpr.atom('gamma',i)
    return reduce_pexpr(base)

def Delta(i,j):
    if i==j: return PExpr.scalar(1)
    return PExpr.atom('Delta',i,j)

def dcat_eq(i,j): return Delta(i,j)

def iscat(cat,t,scalar=None):
    return cat[0]=='s' and cat[1]==scalar if t=='s' else cat[0]==t

def direction_dA2(ireg,dtyp,labels):
    def get(row,col,pool=None):
        rt,ri=row; ct,ci=col
        z=PExpr.zero()
        if dtyp=='X':
            i=labels[0]
            if rt=='s' and ri=='0' and ct=='x': return Delta(ci,i)
            if rt=='x' and ct=='s' and ci=='0': return Delta(ri,i)
            return z
        if dtyp=='XX':
            i,j=labels
            if rt=='x' and ct=='x': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            return z
        if dtyp=='W':
            if rt=='s' and ct=='s' and ((ri,ct,ci)==('0','s','0') or (ri,ct,ci)==('0','s','w') or (ri,ct,ci)==('w','s','0') or (ri,ct,ci)==('w','s','w')):
                return PExpr.one()
            # Above tuple test is awkward; explicit scalar block check below.
            if rt=='s' and ct=='s' and ((ri,ci) in [('0','0'),('0','w'),('w','0'),('w','w')]):
                return PExpr.one()
            return z
        if dtyp=='WX':
            i=labels[0]
            if rt=='x' and ct=='s' and ci=='w': return Delta(ri,i)
            if rt=='s' and ct=='x' and ri=='w': return Delta(ci,i)
            if ireg:
                if rt=='s' and ri=='0' and ct=='wx': return Delta(ci,i)
                if rt=='wx' and ct=='s' and ci=='0': return Delta(ri,i)
                if rt=='s' and ri=='w' and ct=='wx': return Delta(ci,i)
                if rt=='wx' and ct=='s' and ci=='w': return Delta(ri,i)
            return z
        if dtyp=='X2W':
            if not ireg: return z
            i,j=labels
            if rt=='x' and ct=='wx': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            if rt=='wx' and ct=='x': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            if rt=='wx' and ct=='wx': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            return z
        return z
    return get

# Replace the earlier direction builder with delta-aware version.
direction_dA = direction_dA2

def direction_db2(ireg,dtyp,labels):
    def get(cat,pool=None):
        if dtyp=='Y' and cat==cat_scalar('0'): return PExpr.one()
        if dtyp=='XY' and cat[0]=='x': return Delta(cat[1],labels[0])
        if dtyp=='WY' and cat==cat_scalar('w'): return PExpr.one()
        if ireg and dtyp=='XWY' and cat[0]=='wx': return Delta(cat[1],labels[0])
        return PExpr.zero()
    return get

direction_db = direction_db2

def theta0_expr(ireg,cat):
    if cat[0]=='x': return PExpr.atom('gamma',cat[1]).scale(Fraction(1,2))
    if ireg and cat[0]=='wx': return PExpr.atom('gamma',cat[1])
    return PExpr.zero()
theta0 = lambda ireg: (lambda cat: theta0_expr(ireg,cat))

def all_cat_for_dot(ireg,pool):
    return [cat_scalar('0'),cat_scalar('w'),cat_X(pool.fresh())] + ([cat_WX(pool.fresh())] if ireg else [])

def dot_vec(ireg, f, g, pool):
    total=PExpr.zero()
    total += f(cat_scalar('0'))*g(cat_scalar('0'))
    total += f(cat_scalar('w'))*g(cat_scalar('w'))
    q=pool.fresh(); total += f(cat_X(q))*g(cat_X(q))
    if ireg:
        q=pool.fresh(); total += f(cat_WX(q))*g(cat_WX(q))
    return reduce_pexpr(total)

def matvec_any(ireg,A,vfun,row,pool):
    return matvec_generic(ireg,A,vfun,row,pool)

def vec_Rtimes(ireg,vfun,row,pool):
    return matvec_R(ireg,vfun,row,pool)

def dtheta2(ireg,dtyp,labels,pool):
    dA=direction_dA(ireg,dtyp,labels); db=direction_db(ireg,dtyp,labels); th=theta0(ireg)
    def residual(mid):
        return db(mid) - matvec_any(ireg,dA,lambda m:th(m),mid,pool)
    return lambda cat: vec_Rtimes(ireg,residual,cat,pool)

def avec(ireg):
    return lambda cat: Rget(ireg,cat,cat_scalar('w'))

def base_scalar_a(ireg,dtyp,labels,pool):
    dA=direction_dA(ireg,dtyp,labels); db=direction_db(ireg,dtyp,labels); th=theta0(ireg)
    def residual(mid): return db(mid)-matvec_any(ireg,dA,lambda m:th(m),mid,pool)
    val=vec_Rtimes(ireg,residual,cat_scalar('w'),pool)
    if ireg and dtyp=='X': val += PExpr.atom('gamma',labels[0])
    return reduce_pexpr(val)

def elin(ireg,dtyp,labels,cat):
    if ireg and dtyp=='X' and cat[0]=='wx': return Delta(cat[1],labels[0])
    return PExpr.zero()

def endpoint_gamma2(ireg,dtyp,labels,pool):
    # Gamma_of(direction2)[i] = e_aux_i^T dtheta2 = dtheta2(WX_i).
    dt=dtheta2(ireg,dtyp,labels,pool)
    return lambda q: dt(cat_WX(q)) if ireg else PExpr.zero()

def scalar_aT_dA_dt(ireg,dA,dt,pool):
    return dot_vec(ireg,avec(ireg),lambda row: matvec_any(ireg,dA,dt,row,pool),pool)

def scalar_elinT_dt(ireg,dtyp,labels,dt,pool):
    return dot_vec(ireg,lambda cat: elin(ireg,dtyp,labels,cat),dt,pool)

def A3_one(ireg,directions,pool):
    # directions: [(typ,labels), ...]
    (d,dL),(de,deL),(ep,epL)=directions
    dA_d=direction_dA(ireg,d,dL); dA_de=direction_dA(ireg,de,deL); dA_ep=direction_dA(ireg,ep,epL)
    dt_d=dtheta2(ireg,d,dL,pool); dt_de=dtheta2(ireg,de,deL,pool); dt_ep=dtheta2(ireg,ep,epL,pool)
    total=PExpr.zero()
    # -e_lin_d^T R(dA_ep dt_de) - e_lin_d^T R(dA_de dt_ep)
    for da,dt2 in [(dA_ep,dt_de),(dA_de,dt_ep)]:
        v=lambda row: matvec_any(ireg,lambda r,c,p: Rget(ireg,r,c),lambda mid: matvec_any(ireg,da,dt2,mid,pool),row,pool)
        total -= scalar_elinT_dt_v(ireg,d, dL, v, pool)
    # Remaining terms are a^T dA_i R(dA_j dt_k), with signs as in frozen formula.
    terms=[
      (+1,dA_ep,dA_de,dt_d),
      (-1,None,None,None), # placeholder below matches exact 12-term list
    ]
    # Direct transcription of the frozen 12-term formula, each written as
    # a^T A1 R(A2 dt3) or -e_lin^T R(A2 dt3).
    def a_A_R_A_dt(A1,A2,dt):
        inner=lambda mid: matvec_any(ireg,A2,dt,mid,pool)
        rinner=lambda row: matvec_any(ireg,lambda r,c,p:Rget(ireg,r,c),inner,row,pool)
        return dot_vec(ireg,avec(ireg),lambda row: matvec_any(ireg,A1,rinner,row,pool),pool)
    def e_R_A_dt(typ,L,A2,dt):
        inner=lambda mid: matvec_any(ireg,A2,dt,mid,pool)
        rinner=lambda row: matvec_any(ireg,lambda r,c,p:Rget(ireg,r,c),inner,row,pool)
        return dot_vec(ireg,lambda cat: elin(ireg,typ,L,cat),rinner,pool)
    total=PExpr.zero()
    total -= e_R_A_dt(d,dL,dA_ep,dt_de)
    total -= e_R_A_dt(d,dL,dA_de,dt_ep)
    total += a_A_R_A_dt(dA_ep,dA_de,dt_d)
    total -= e_R_A_dt(ep,epL,dA_de,dt_d)
    total += a_A_R_A_dt(dA_de,dA_ep,dt_d)
    total += a_A_R_A_dt(dA_de,dA_d,dt_ep)
    total -= e_R_A_dt(de,deL,dA_ep,dt_d)
    total -= e_R_A_dt(de,deL,dA_d,dt_ep)
    total += a_A_R_A_dt(dA_ep,dA_d,dt_de)
    total -= e_R_A_dt(ep,epL,dA_d,dt_de)
    total += a_A_R_A_dt(dA_d,dA_ep,dt_de)
    total += a_A_R_A_dt(dA_d,dA_de,dt_ep)
    return reduce_pexpr(total)

def scalar_elinT_dt_v(ireg,dtyp,labels,v,pool):
    return dot_vec(ireg,lambda cat: elin(ireg,dtyp,labels,cat),v,pool)

# Coordinate-type specifications and generic raw moments.

def spec_poly(kind, labels, gamma_cache=None, pool=None):
    # Random polynomial as sparse dict[(d,e,tuple(X labels))] -> coefficient PExpr.
    # coefficient is a PExpr in gamma tensor atoms.
    # Y = (delta+1/2) sum_a gamma_a X_a + eps; W = delta+1/2.
    def add(A,B):
        C=dict(A)
        for k,v in B.items():
            if v: C[k]=C.get(k,PExpr.zero())+v
        return {k:v for k,v in C.items() if v}
    def mul(A,B):
        C={}
        for (d1,e1,x1),c1 in A.items():
          for (d2,e2,x2),c2 in B.items():
            key=(d1+d2,e1+e2,tuple(x1+x2)); C[key]=C.get(key,PExpr.zero())+c1*c2
        return C
    one={(0,0,()):PExpr.one()}
    W={(1,0,()):PExpr.one(),(0,0,()):PExpr.scalar(Fraction(1,2))}
    def X(i): return {(0,0,(i,)):PExpr.one()}
    Y={}
    # delta gamma'X + 1/2 gamma'X + eps
    q=pool.fresh() if pool else None
    # For a generic Y, the internal gamma index must be summed, represented by a
    # fresh repeated index.  The gamma label appears once; expectation tensor will
    # contract it only when the X label is the same dummy. This construction creates
    # an explicit dummy label in each Y and must be canonicalized at term level.
    a=q if q is not None else -1
    Y[(1,0,(a,))]=PExpr.atom('gamma',a)
    Y[(0,0,(a,))]=PExpr.atom('gamma',a).scale(Fraction(1,2))
    Y[(0,1,())]=PExpr.one()
    if kind=='X': return X(labels[0])
    if kind=='XX': return mul(X(labels[0]),X(labels[1]))
    if kind=='W': return W
    if kind=='WX': return mul(W,X(labels[0]))
    if kind=='Y': return Y
    if kind=='XY': return mul(X(labels[0]),Y)
    if kind=='WY': return mul(W,Y)
    if kind=='X2W': return mul(mul(X(labels[0]),X(labels[1])),W)
    if kind=='XWY': return mul(mul(X(labels[0]),W),Y)
    raise ValueError(kind)

def expect_poly(poly):
    # Reduce delta powers and map X^deg eps^e to primitive tensor atoms.
    out=PExpr.zero()
    for (d,e,xs),coeff in poly.items():
        q,drem=divmod(d,2); factor=sp.Rational(1,4)**q
        if drem==1:
            # d=1 moments: zero for e=0; zero for e=1 deg<=1; otherwise P/Z/etc.
            pass
        deg=len(xs)
        xs=tuple(xs)
        if drem==1 and e==0: continue
        if drem==1 and e==1 and deg<=1: continue
        if drem==0 and e==1 and deg<=1: continue
        if e>=3:
            atom=PExpr.atom(f'NEW_{deg}_{drem}_{e}',*xs)
        elif deg==0 and drem==0 and e==0: atom=PExpr.one()
        elif deg==0 and drem==0 and e==1: continue
        elif deg==0 and drem==0 and e==2: atom=PExpr.atom('sigma_eps2')
        elif deg==0 and drem==1 and e==2: atom=PExpr.atom('rho')
        elif deg==1 and drem==0 and e==2: atom=PExpr.atom('N',xs[0])
        elif deg==2 and drem==0 and e==0: atom=PExpr.atom('Sigma',*xs)
        elif deg==2 and drem==0 and e==1: atom=PExpr.atom('M',*xs)
        elif deg==2 and drem==0 and e==2: atom=PExpr.atom('R',*xs)
        elif deg==2 and drem==1 and e==1: atom=PExpr.atom('P',*xs)
        elif deg==3 and drem==0 and e==0: atom=PExpr.atom('T3',*xs)
        elif deg==3 and drem==0 and e==1: atom=PExpr.atom('S',*xs)
        elif deg==3 and drem==1 and e==1: atom=PExpr.atom('Z',*xs)
        elif deg==4 and drem==0 and e==0: atom=PExpr.atom('T4',*xs)
        elif deg==4 and drem==1 and e==0: continue
        else: atom=PExpr.atom(f'NEW_{deg}_{drem}_{e}',*xs)
        out += coeff*atom.scale(factor)
    return reduce_pexpr(out)

def moment_m2(t1,L1,t2,L2):
    return expect_poly(spec_mul(spec_poly(t1,L1),spec_poly(t2,L2)))

def spec_mul(A,B):
    C={}
    for (d1,e1,x1),c1 in A.items():
      for (d2,e2,x2),c2 in B.items():
        k=(d1+d2,e1+e2,tuple(x1+x2)); C[k]=C.get(k,PExpr.zero())+c1*c2
    return C

def moment_m3(t1,L1,t2,L2,t3,L3):
    return expect_poly(spec_mul(spec_mul(spec_poly(t1,L1,pool=IndexPool()),spec_poly(t2,L2,pool=IndexPool())),spec_poly(t3,L3,pool=IndexPool())))

def H_one(ireg, d1, L1, d2, L2, pool):
    A1=direction_dA(ireg,d1,L1); A2=direction_dA(ireg,d2,L2)
    dt1=dtheta2(ireg,d1,L1,pool); dt2=dtheta2(ireg,d2,L2,pool)
    total=PExpr.zero()
    if ireg and d1=='X':
        total += dt2(cat_WX(L1[0]))
    # -a^T A2 dt1 + e_lin2^T dt1 - a^T A1 dt2
    total -= scalar_aT_dA_dt(ireg,A2,dt1,pool)
    total += scalar_elinT_dt(ireg,d2,L2,dt1,pool)
    total -= scalar_aT_dA_dt(ireg,A1,dt2,pool)
    return reduce_pexpr(total)

# Coordinate types and their index arity.
COORD_TYPES_R=['X','XX','W','WX','Y','XY','WY']
COORD_TYPES_I=COORD_TYPES_R+['X2W','XWY']
ARITY={'X':1,'XX':2,'W':0,'WX':1,'Y':0,'XY':1,'WY':0,'X2W':2,'XWY':1}
SYMPAIR={'XX','X2W'}

# Rename every index label in a PExpr, used to turn cached generic derivative/moment
# expressions into fresh instances in a complete Hall term.
def rename_pexpr(e,mapping):
    d=defaultdict(lambda:sp.Integer(0))
    for fs,c in e.terms.items():
        nfs=[]
        for f in fs:
            nfs.append(Factor(f.name,tuple(mapping.get(i,i) for i in f.inds)))
        d[tuple(nfs)] += c
    return PExpr(d).local_sym()

def generic_labels(n): return list(range(n))

class CachedSymbolics:
    def __init__(self,ireg):
        self.ireg=ireg; self.pool=IndexPool(); self.deriv={}; self.mom2={}; self.mom3={}
    def deriv_labels(self,directions):
        # directions are (type, arity); generic coordinate labels occupy 0..A-1.
        total=sum(ARITY[t] for t,_ in directions)
        # pool starts above all coordinate labels.
        self.pool=IndexPool(); self.pool.next=total+10
        off=0; inst=[]
        for typ,arity in directions:
            labs=list(range(off,off+arity)); off+=arity; inst.append((typ,labs))
        return inst
    def get_a(self,typ):
        key=typ
        if key not in self.deriv:
            self.pool=IndexPool(); self.pool.next=ARITY[typ]+10
            labs=list(range(ARITY[typ]))
            self.deriv[key]=a_direction(self.ireg,typ,labs,self.pool)
        return self.deriv[key]
    def get_H(self,t1,t2):
        key=(t1,t2)
        if key not in self.deriv:
            inst=self.deriv_labels([(t1,ARITY[t1]),(t2,ARITY[t2])]); (d1,L1),(d2,L2)=inst
            self.deriv[key]=H_one(self.ireg,d1,L1,d2,L2,self.pool)
        return self.deriv[key]
    def get_A3(self,t1,t2,t3):
        key=('A3',t1,t2,t3)
        if key not in self.deriv:
            inst=self.deriv_labels([(t1,ARITY[t1]),(t2,ARITY[t2]),(t3,ARITY[t3])])
            self.deriv[key]=A3_one(self.ireg,inst,self.pool)
        return self.deriv[key]
    def get_m2(self,t1,t2):
        # generic coordinate labels 0..arity-1, then second coordinate labels offset.
        key=('m2',t1,t2)
        if key not in self.mom2:
            off1=0; off2=ARITY[t1]; labs1=list(range(off1,off1+ARITY[t1])); labs2=list(range(off2,off2+ARITY[t2]))
            pool=IndexPool(); pool.next=off2+ARITY[t2]+10
            e=spec_mul(spec_poly(t1,labs1,pool=pool),spec_poly(t2,labs2,pool=pool))
            self.mom2[key]=expect_poly(e)
        return self.mom2[key]
    def get_m3(self,t1,t2,t3):
        key=(t1,t2,t3)
        if key not in self.mom3:
            off1=0; off2=ARITY[t1]; off3=off2+ARITY[t2]
            labs1=list(range(off1,off1+ARITY[t1])); labs2=list(range(off2,off2+ARITY[t2])); labs3=list(range(off3,off3+ARITY[t3]))
            pool=IndexPool(); pool.next=off3+ARITY[t3]+10
            e=spec_mul(spec_mul(spec_poly(t1,labs1,pool=pool),spec_poly(t2,labs2,pool=pool)),spec_poly(t3,labs3,pool=pool))
            E123=expect_poly(e)
            E1=expect_poly(spec_poly(t1,labs1,pool=pool)); E2=expect_poly(spec_poly(t2,labs2,pool=pool)); E3=expect_poly(spec_poly(t3,labs3,pool=pool))
            E12=expect_poly(spec_mul(spec_poly(t1,labs1,pool=pool),spec_poly(t2,labs2,pool=pool)))
            E13=expect_poly(spec_mul(spec_poly(t1,labs1,pool=pool),spec_poly(t3,labs3,pool=pool)))
            E23=expect_poly(spec_mul(spec_poly(t2,labs2,pool=pool),spec_poly(t3,labs3,pool=pool)))
            self.mom3[key]=canonicalize(reduce_pexpr(E123-E1*E23-E2*E13-E3*E12+2*E1*E2*E3))
        return self.mom3[key]

def instantiate(e:PExpr, coord_actual, total_coord_labels, pool:IndexPool):
    """Instantiate a cached generic-index expression.
    coord_actual is a dict {generic_coordinate_label: actual_label}.
    Every generic bound/internal label not in the coordinate set receives a fresh
    label. This is alpha-renaming only.
    """
    generic_labels=set(range(total_coord_labels))
    all_labels=sorted(set(i for fs in e.terms for f in fs for i in f.inds))
    mp=dict(coord_actual)
    for old in all_labels:
        if old not in generic_labels and old not in mp:
            mp[old]=pool.fresh()
    return rename_pexpr(e,mp)

def m1_cached(cacher,typ,L,pool):
    ar=ARITY[typ]; generic=list(range(ar)); total=ar
    # build fresh cache expression directly from spec with generic labels
    pool2=IndexPool(); pool2.next=ar+10
    return instantiate(expect_poly(spec_poly(typ,generic,pool=pool2)), {i:L[i] for i in range(ar)}, ar, pool)

def m2_cached(cacher,t1,L1,t2,L2,pool):
    e=cacher.get_m2(t1,t2); ar1=ARITY[t1]; ar2=ARITY[t2]; total=ar1+ar2
    mp={i:L1[i] for i in range(ar1)}
    mp.update({ar1+i:L2[i] for i in range(ar2)})
    return instantiate(e,mp,total,pool)

def m3_cached(cacher,t1,L1,t2,L2,t3,L3,pool):
    e=cacher.get_m3(t1,t2,t3); ar1=ARITY[t1]; ar2=ARITY[t2]; ar3=ARITY[t3]; total=ar1+ar2+ar3
    mp={i:L1[i] for i in range(ar1)}
    mp.update({ar1+i:L2[i] for i in range(ar2)})
    mp.update({ar1+ar2+i:L3[i] for i in range(ar3)})
    return instantiate(e,mp,total,pool)

def cov_cached(cacher,t1,L1,t2,L2,pool):
    return m2_cached(cacher,t1,L1,t2,L2,pool) - m1_cached(cacher,t1,L1,pool)*m1_cached(cacher,t2,L2,pool)

def cum3_cached(cacher,t1,L1,t2,L2,t3,L3,pool):
    e123=m3_cached(cacher,t1,L1,t2,L2,t3,L3,pool)
    e1=m1_cached(cacher,t1,L1,pool); e2=m1_cached(cacher,t2,L2,pool); e3=m1_cached(cacher,t3,L3,pool)
    e12=m2_cached(cacher,t1,L1,t2,L2,pool); e13=m2_cached(cacher,t1,L1,t3,L3,pool); e23=m2_cached(cacher,t2,L2,t3,L3,pool)
    return reduce_pexpr(e123-e1*e23-e2*e13-e3*e12+2*e1*e2*e3)

def coord_branches(typ,pool):
    """One coordinate occurrence as exact sum over its feature index/indices.
    For symmetric pair coordinates, return the 1/2 ordered sum branch and the
    1/2 diagonal branch."""
    ar=ARITY[typ]
    if ar==0: return [(typ,(),sp.Rational(1))]
    if typ in SYMPAIR:
        i=pool.fresh(); j=pool.fresh()
        return [(typ,(i,j),sp.Rational(1,2)), (typ,(i,i),sp.Rational(1,2))]
    return [(typ,(pool.fresh(),),sp.Rational(1))]

def coord_combinations(types,pool):
    # Cartesian product, preserving exact symmetric-pair decomposition.
    out=[((),sp.Integer(1))]
    for typ in types:
        branches=coord_branches(typ,pool)
        nxt=[]
        for cur,w in out:
            for t,L,bw in branches: nxt.append((cur+((t,L),),w*bw))
        out=nxt
    return out

# Cached scalar derivative instances. Each call returns an expression in actual
# coordinate labels and freshly alpha-renamed bound labels.
def a_inst(cacher,typ,L,pool):
    e=cacher.get_a(typ); return instantiate(e,{i:L[i] for i in range(ARITY[typ])},ARITY[typ],pool)
def H_inst(cacher,t1,L1,t2,L2,pool):
    e=cacher.get_H(t1,t2); ar1=ARITY[t1]; ar2=ARITY[t2]
    mp={i:L1[i] for i in range(ar1)}; mp.update({ar1+i:L2[i] for i in range(ar2)})
    return instantiate(e,mp,ar1+ar2,pool)
def A3_inst(cacher,t1,L1,t2,L2,t3,L3,pool):
    e=cacher.get_A3(t1,t2,t3); a1=ARITY[t1]; a2=ARITY[t2]; a3=ARITY[t3]
    mp={i:L1[i] for i in range(a1)}; mp.update({a1+i:L2[i] for i in range(a2)}); mp.update({a1+a2+i:L3[i] for i in range(a3)})
    return instantiate(e,mp,a1+a2+a3,pool)

def hall_indexed(ireg=True, restrict_types=None, verbose=True):
    cacher=CachedSymbolics(ireg)
    types=list(restrict_types or (COORD_TYPES_I if ireg else COORD_TYPES_R))
    # We cache derivative objects up front lazily. Hall terms use complete type Cartesian products.
    out=[PExpr.zero(),PExpr.zero(),PExpr.zero()]
    stats={}
    # T1
    t0=__import__('time').time(); total=0
    # a-zero screening is algebraically exact because get_a(type) is exact.
    for (c1,w1) in coord_combinations(types,IndexPool()):
        pass
    # We need a separate pool per full term so no unrelated dummy labels collide.
    for t1 in types:
      # determine generic a nonzero cheaply; if zero, entire t1 contribution zero.
      probe=CachedSymbolics(ireg).get_a(t1)
      if not probe.terms: continue
      for t2 in types:
       for t3 in types:
        # H may be identically zero; exact symbolic computation once.
        Hp= c=cacher.get_H(t2,t3)
        if not Hp.terms: continue
        # symmetric-pair branch combinations
        base_types=[t1,t2,t3]
        pool=IndexPool(); branches=coord_combinations(base_types,pool)
        for insts,w in branches:
            (T1,L1),(T2,L2),(T3,L3)=insts
            # instantiate all with one term pool; derivative caches have their own generic labels.
            term=a_inst(cacher,T1,L1,pool)*H_inst(cacher,T2,L2,T3,L3,pool)
            term=term*cum3_cached(cacher,T1,L1,T2,L2,T3,L3,pool)
            out[0]+=term.scale(w); total+=1
    stats['T1_terms']=total
    if verbose: print('T1 branches',total,'terms',out[0].num_terms(),flush=True)
    # T2: four coordinate directions.
    total=0
    for t1 in types:
      for t2 in types:
       Hp=cacher.get_H(t1,t2)
       if not Hp.terms: continue
       for t3 in types:
        for t4 in types:
         Hq=cacher.get_H(t3,t4)
         if not Hq.terms: continue
         pool=IndexPool(); branches=coord_combinations([t1,t2,t3,t4],pool)
         for insts,w in branches:
            (A,LA),(B,LB),(C,LC),(D,LD)=insts
            h1=H_inst(cacher,A,LA,B,LB,pool); h2=H_inst(cacher,C,LC,D,LD,pool)
            cAC=cov_cached(cacher,A,LA,C,LC,pool); cBD=cov_cached(cacher,B,LB,D,LD,pool)
            cAD=cov_cached(cacher,A,LA,D,LD,pool); cBC=cov_cached(cacher,B,LB,C,LC,pool)
            cAB=cov_cached(cacher,A,LA,B,LB,pool); cCD=cov_cached(cacher,C,LC,D,LD,pool)
            wick=cAC*cBD+cAD*cBC+cAB*cCD
            out[1]+=h1*h2*wick.scale(w*sp.Rational(1,4)); total+=1
    stats['T2_terms']=total
    if verbose: print('T2 branches',total,'terms',out[1].num_terms(),flush=True)
    # T3: four directions a[n1] A3[n2,n3,n4] and Wick pairings.
    total=0
    for t1 in types:
      ap=cacher.get_a(t1)
      if not ap.terms: continue
      for t2 in types:
       for t3 in types:
        for t4 in types:
         a3p=cacher.get_A3(t2,t3,t4)
         if not a3p.terms: continue
         pool=IndexPool(); branches=coord_combinations([t1,t2,t3,t4],pool)
         for insts,w in branches:
            (A,LA),(B,LB),(C,LC),(D,LD)=insts
            av=a_inst(cacher,A,LA,pool); a3=A3_inst(cacher,B,LB,C,LC,D,LD,pool)
            cAB=cov_cached(cacher,A,LA,B,LB,pool); cCD=cov_cached(cacher,C,LC,D,LD,pool)
            cAC=cov_cached(cacher,A,LA,C,LC,pool); cBD=cov_cached(cacher,B,LB,D,LD,pool)
            cAD=cov_cached(cacher,A,LA,D,LD,pool); cBC=cov_cached(cacher,B,LB,C,LC,pool)
            wick=cAB*cCD+cAC*cBD+cAD*cBC
            out[2]+=av*a3*wick.scale(w*sp.Rational(1,3)); total+=1
    stats['T3_terms']=total
    if verbose: print('T3 branches',total,'terms',out[2].num_terms(),flush=True)
    return tuple(canonicalize(reduce_pexpr(x)) for x in out),cacher,stats

# Invariant target in the same indexed algebra, with p symbolic.
def target_general_p():
    # Use fresh labels but expressions are closed scalars; every index is summed by repetition.
    pool=IndexPool(); i,j,k,l,a,b,c,d=pool.many(8)
    g=lambda q:PExpr.atom('gamma',q)
    G=PExpr.atom('gamma',i)*PExpr.atom('Sigma',i,j)*PExpr.atom('gamma',j)
    F=PExpr.atom('gamma',i)*PExpr.atom('gamma',j)*PExpr.atom('T4',i,j,k,l)*PExpr.atom('K',k,l)
    qj=PExpr.atom('T3',j,k,l)*PExpr.atom('K',k,l)
    Q=(PExpr.atom('gamma',j)*qj)*(PExpr.atom('gamma',i)*PExpr.atom('T3',i,a,b)*PExpr.atom('K',a,b))
    # Above Q is duplicate construction; simplify to (gamma^T q)^2 by a direct two-copy product.
    h=PExpr.atom('gamma',i)*PExpr.atom('T3',i,k,l)*PExpr.atom('K',k,l)
    h2=PExpr.atom('gamma',a)*PExpr.atom('T3',a,b,c)*PExpr.atom('K',b,c)
    Q=h*h2
    R=PExpr.atom('gamma',i)*PExpr.atom('gamma',j)*PExpr.atom('T3',i,a,b)*PExpr.atom('K',a,c)*PExpr.atom('K',b,d)*PExpr.atom('T3',j,c,d)
    AP=PExpr.atom('gamma',i)*PExpr.atom('T3',i,j,k)*PExpr.atom('K',j,a)*PExpr.atom('P',a,b)*PExpr.atom('K',b,k)
    BP=h * (PExpr.atom('K',c,d)*PExpr.atom('P',d,c))
    CP=PExpr.atom('gamma',i)*PExpr.atom('P',i,a)*PExpr.atom('K',a,j)*PExpr.atom('T3',j,k,l)*PExpr.atom('K',k,l)
    UM=PExpr.atom('K',i,j)*PExpr.atom('M',j,k)*PExpr.atom('K',k,l)*PExpr.atom('M',l,i)
    UN=PExpr.atom('T3',i,j,k)*PExpr.atom('K',j,k)*PExpr.atom('K',i,l)*PExpr.atom('N',l)
    p=sp.Symbol('p')
    # Row-sum target for D = -(2p+7)G + F-Q-3R -16AP-8BP+8CP+8UM+8UN.
    return reduce_pexpr(G.scale(-(2*p+7))+F-Q-3*R-16*AP-8*BP+8*CP+8*UM+8*UN)

# Fast exact closed-form population derivative directions.  These are obtained by
# multiplying the displayed block inverse R by each sparse dA/db entry, with no
# coordinate instantiation.  They are used only as an optimization of the same
# indexed matrix algebra; the sparse dA objects above remain the source of truth.
def direct_dt(ireg,dtyp,labels):
    def v(cat):
        t,i=cat
        z=PExpr.zero()
        if not ireg:
            if dtyp=='X':
                return PExpr.zero()  # feature components zero; scalar handled below
            if dtyp=='XX':
                if t=='x': return (PExpr.atom('K',i,labels[0]).scale(-sp.Rational(1,2))*PExpr.atom('gamma',labels[1])
                                      +PExpr.atom('K',i,labels[1]).scale(-sp.Rational(1,2))*PExpr.atom('gamma',labels[0]))
                return PExpr.zero()
            if dtyp=='WX':
                return PExpr.zero()
            if dtyp=='XY' and t=='x': return PExpr.atom('K',i,labels[0])
            return PExpr.zero()
        else:
            if dtyp=='X' or dtyp=='XX': return z
            if dtyp=='X2W' and t=='wx':
                return PExpr.atom('K',i,labels[0]).scale(-2)*PExpr.atom('gamma',labels[1]) + PExpr.atom('K',i,labels[1]).scale(-2)*PExpr.atom('gamma',labels[0])
            if dtyp=='XY' and t=='x': return PExpr.atom('K',i,labels[0]).scale(2)
            if dtyp=='XWY' and t=='x': return PExpr.atom('K',i,labels[0]).scale(-2)
            if dtyp=='XWY' and t=='wx': return PExpr.atom('K',i,labels[0]).scale(4)
            return z
    # Scalar components are returned below by a wrapper.
    def out(cat):
        t,i=cat
        if not ireg:
            if dtyp=='X':
                if t=='s' and i=='0': return PExpr.atom('gamma',labels[0]).scale(-1)
                if t=='s' and i=='w': return PExpr.atom('gamma',labels[0])
            if dtyp=='WX':
                if t=='s' and i=='0': return PExpr.atom('gamma',labels[0])
                if t=='s' and i=='w': return PExpr.atom('gamma',labels[0]).scale(-2)
            if dtyp=='Y':
                if t=='s' and i=='0': return PExpr.scalar(2)
                if t=='s' and i=='w': return PExpr.scalar(-2)
            if dtyp=='WY':
                if t=='s' and i=='0': return PExpr.scalar(-2)
                if t=='s' and i=='w': return PExpr.scalar(4)
            return v(cat)
        else:
            if dtyp=='WX':
                if t=='s' and i=='0': return PExpr.atom('gamma',labels[0]).scale(2) * PExpr.scalar(0)  # overwritten below
                if t=='s' and i=='w': return PExpr.atom('gamma',labels[0]).scale(-2)
                # scalar intercept is zero
            if dtyp=='Y':
                if t=='s' and i=='0': return PExpr.scalar(2)
                if t=='s' and i=='w': return PExpr.scalar(-2)
            if dtyp=='WY':
                if t=='s' and i=='0': return PExpr.scalar(-2)
                if t=='s' and i=='w': return PExpr.scalar(4)
            return v(cat)
    # Correct IREG WX scalar components: theta0=0, thetaW=-2 gamma.
    return out

# Correct typo-friendly wrapper; defining this separately keeps all formulas explicit.
def dtheta2(ireg,dtyp,labels,pool):
    direct=direct_dt(ireg,dtyp,labels)
    return direct

def a_direction(ireg,dtyp,labels,pool):
    if ireg:
        if dtyp=='X': return PExpr.atom('gamma',labels[0])
        if dtyp=='WX': return PExpr.atom('gamma',labels[0]).scale(-2)
        if dtyp=='Y': return PExpr.scalar(-2)
        if dtyp=='WY': return PExpr.scalar(4)
        return PExpr.zero()
    else:
        if dtyp=='X': return PExpr.atom('gamma',labels[0])
        if dtyp=='WX': return PExpr.atom('gamma',labels[0]).scale(-2)
        if dtyp=='Y': return PExpr.scalar(-2)
        if dtyp=='WY': return PExpr.scalar(4)
        return PExpr.zero()

# Fast generic cumulant/covariance caches: build the complete indexed object once
# per coordinate-type tuple, then only alpha-rename it for each Hall summand.
CachedSymbolics._cov_expr_cache = {}
CachedSymbolics._cum3_expr_cache = {}
def get_cov_expr(self,t1,t2):
    key=(id(self),t1,t2)
    cache=getattr(self,'_cov_local',None)
    if cache is None: self._cov_local={}; cache=self._cov_local
    if (t1,t2) not in cache:
        ar1=ARITY[t1]; ar2=ARITY[t2]
        l1=list(range(ar1)); l2=list(range(ar1,ar1+ar2))
        p=IndexPool(); p.next=ar1+ar2+10
        E2=expect_poly(spec_mul(spec_poly(t1,l1,pool=p),spec_poly(t2,l2,pool=p)))
        E1=expect_poly(spec_poly(t1,l1,pool=p)); E22=expect_poly(spec_poly(t2,l2,pool=p))
        cache[(t1,t2)]=reduce_pexpr(E2-E1*E22)
    return cache[(t1,t2)]
def get_cum3_expr(self,t1,t2,t3):
    cache=getattr(self,'_cum3_local',None)
    if cache is None: self._cum3_local={}; cache=self._cum3_local
    key=(t1,t2,t3)
    if key not in cache:
        a1,a2,a3=ARITY[t1],ARITY[t2],ARITY[t3]
        l1=list(range(a1)); l2=list(range(a1,a1+a2)); l3=list(range(a1+a2,a1+a2+a3))
        p=IndexPool(); p.next=a1+a2+a3+10
        e1=expect_poly(spec_poly(t1,l1,pool=p)); e2=expect_poly(spec_poly(t2,l2,pool=p)); e3=expect_poly(spec_poly(t3,l3,pool=p))
        e12=expect_poly(spec_mul(spec_poly(t1,l1,pool=p),spec_poly(t2,l2,pool=p)))
        e13=expect_poly(spec_mul(spec_poly(t1,l1,pool=p),spec_poly(t3,l3,pool=p)))
        e23=expect_poly(spec_mul(spec_poly(t2,l2,pool=p),spec_poly(t3,l3,pool=p)))
        e123=expect_poly(spec_mul(spec_mul(spec_poly(t1,l1,pool=p),spec_poly(t2,l2,pool=p)),spec_poly(t3,l3,pool=p)))
        cache[key]=reduce_pexpr(e123-e1*e23-e2*e13-e3*e12+2*e1*e2*e3)
    return cache[key]
CachedSymbolics.get_cov_expr=get_cov_expr; CachedSymbolics.get_cum3_expr=get_cum3_expr

def cov_fast(cacher,t1,L1,t2,L2,pool):
    e=cacher.get_cov_expr(t1,t2); a1=ARITY[t1]; a2=ARITY[t2]
    mp={i:L1[i] for i in range(a1)}; mp.update({a1+i:L2[i] for i in range(a2)})
    return instantiate(e,mp,a1+a2,pool)
def cum3_fast(cacher,t1,L1,t2,L2,t3,L3,pool):
    e=cacher.get_cum3_expr(t1,t2,t3); a1=ARITY[t1]; a2=ARITY[t2]; a3=ARITY[t3]
    mp={i:L1[i] for i in range(a1)}; mp.update({a1+i:L2[i] for i in range(a2)}); mp.update({a1+a2+i:L3[i] for i in range(a3)})
    return instantiate(e,mp,a1+a2+a3,pool)

# Fast exact canonicalization for closed scalar terms.  Dummy-index equality is
# graph isomorphism of the incidence graph; factor names and tensor-slot labels
# are retained, while index nodes themselves are unlabeled.  This is an
# implementation detail for alpha-renaming, not a mathematical contraction
# classification or an assumption about which tensors may occur.
def _term_graph(factors):
    import networkx as nx
    G=nx.Graph()
    for fi,f in enumerate(factors):
        fn=f'F{fi}'; G.add_node(fn,bip='F',name=f.name)
        for s,i in enumerate(f.inds):
            inn=f'I{i}'
            if inn not in G: G.add_node(inn,bip='I',name='I')
            G.add_edge(fn,inn,slot=s)
    return G

def _wl_key(factors):
    import networkx as nx
    G=_term_graph(factors)
    return nx.weisfeiler_lehman_graph_hash(G,node_attr='name',edge_attr='slot')

def _iso(f1,f2):
    import networkx as nx
    G1=_term_graph(f1); G2=_term_graph(f2)
    nm=nx.algorithms.isomorphism.categorical_node_match('name',None)
    em=nx.algorithms.isomorphism.categorical_edge_match('slot',None)
    return nx.is_isomorphic(G1,G2,node_match=nm,edge_match=em)

def canonicalize(e:PExpr):
    import networkx as nx  # imported here so ordinary algebra has no dependency overhead
    buckets={}
    reps=[]
    for fs,c in e.terms.items():
        fs2=[]
        for f in fs:
            inds=tuple(sorted(f.inds)) if f.name in SYMMETRIC else f.inds
            fs2.append(Factor(f.name,inds))
        fs2=tuple(sorted(fs2,key=lambda f:(f.name,f.inds)))
        h=_wl_key(fs2)
        bucket=buckets.setdefault(h,[])
        found=False
        for idx in bucket:
            if _iso(reps[idx][0],fs2):
                reps[idx]=(reps[idx][0],sp.expand(reps[idx][1]+c)); found=True; break
        if not found:
            bucket.append(len(reps)); reps.append((fs2,sp.expand(c)))
    d=defaultdict(lambda:sp.Integer(0))
    for fs,c in reps:
        if c!=0: d[fs]+=c
    return PExpr(d)

# Patch expectation zero for centered X first moments.
_old_expect_poly = expect_poly
def expect_poly(poly):
    out=PExpr.zero()
    for (d,e,xs),coeff in poly.items():
        q,drem=divmod(d,2); factor=sp.Rational(1,4)**q
        deg=len(xs)
        if drem==1 and e==0: continue
        if drem==0 and e==0 and deg==1: continue
        if drem==1 and e==1 and deg<=1: continue
        if drem==0 and e==1 and deg<=1: continue
        if e>=3: atom=PExpr.atom(f'NEW_{deg}_{drem}_{e}',*xs)
        elif deg==0 and drem==0 and e==0: atom=PExpr.one()
        elif deg==0 and drem==0 and e==1: continue
        elif deg==0 and drem==0 and e==2: atom=PExpr.atom('sigma_eps2')
        elif deg==0 and drem==1 and e==2: atom=PExpr.atom('rho')
        elif deg==1 and drem==0 and e==2: atom=PExpr.atom('N',xs[0])
        elif deg==2 and drem==0 and e==0: atom=PExpr.atom('Sigma',*xs)
        elif deg==2 and drem==0 and e==1: atom=PExpr.atom('M',*xs)
        elif deg==2 and drem==0 and e==2: atom=PExpr.atom('R',*xs)
        elif deg==2 and drem==1 and e==1: atom=PExpr.atom('P',*xs)
        elif deg==3 and drem==0 and e==0: atom=PExpr.atom('T3',*xs)
        elif deg==3 and drem==0 and e==1: atom=PExpr.atom('S',*xs)
        elif deg==3 and drem==1 and e==1: atom=PExpr.atom('Z',*xs)
        elif deg==4 and drem==0 and e==0: atom=PExpr.atom('T4',*xs)
        elif deg==4 and drem==1 and e==0: continue
        else: atom=PExpr.atom(f'NEW_{deg}_{drem}_{e}',*xs)
        out += coeff*atom.scale(factor)
    return reduce_pexpr(out)

def direct_dt(ireg,dtyp,labels):
    def out(cat):
        t,i=cat
        if not ireg:
            if dtyp=='X':
                if t=='s' and i=='0': return PExpr.atom('gamma',labels[0]).scale(-1)
                if t=='s' and i=='w': return PExpr.atom('gamma',labels[0])
            if dtyp=='XX' and t=='x':
                return PExpr.atom('K',i,labels[0]).scale(-sp.Rational(1,2))*PExpr.atom('gamma',labels[1]) + PExpr.atom('K',i,labels[1]).scale(-sp.Rational(1,2))*PExpr.atom('gamma',labels[0])
            if dtyp=='WX':
                if t=='s' and i=='0': return PExpr.atom('gamma',labels[0])
                if t=='s' and i=='w': return PExpr.atom('gamma',labels[0]).scale(-2)
            if dtyp=='Y':
                if t=='s' and i=='0': return PExpr.scalar(2)
                if t=='s' and i=='w': return PExpr.scalar(-2)
            if dtyp=='XY' and t=='x': return PExpr.atom('K',i,labels[0])
            if dtyp=='WY':
                if t=='s' and i=='0': return PExpr.scalar(-2)
                if t=='s' and i=='w': return PExpr.scalar(4)
            return PExpr.zero()
        # IREG feature block inverse [[2K,-2K],[-2K,4K]], scalar block [[2,-2],[-2,4]].
        if dtyp=='WX':
            if t=='s' and i=='w': return PExpr.atom('gamma',labels[0]).scale(-2)
        if dtyp=='Y':
            if t=='s' and i=='0': return PExpr.scalar(2)
            if t=='s' and i=='w': return PExpr.scalar(-2)
        if dtyp=='WY':
            if t=='s' and i=='0': return PExpr.scalar(-2)
            if t=='s' and i=='w': return PExpr.scalar(4)
        if dtyp=='XY':
            if t=='x': return PExpr.atom('K',i,labels[0]).scale(2)
            if t=='wx': return PExpr.atom('K',i,labels[0]).scale(-2)
        if dtyp=='XWY':
            if t=='x': return PExpr.atom('K',i,labels[0]).scale(-2)
            if t=='wx': return PExpr.atom('K',i,labels[0]).scale(4)
        if dtyp=='X2W' and t=='wx':
            return PExpr.atom('K',i,labels[0]).scale(-2)*PExpr.atom('gamma',labels[1]) + PExpr.atom('K',i,labels[1]).scale(-2)*PExpr.atom('gamma',labels[0])
        return PExpr.zero()
    return out

def direction_dA(ireg,dtyp,labels):
    def get(row,col,pool=None):
        rt,ri=row; ct,ci=col
        z=PExpr.zero()
        if dtyp=='X':
            i=labels[0]
            if rt=='s' and ri=='0' and ct=='x': return Delta(ci,i)
            if rt=='x' and ct=='s' and ci=='0': return Delta(ri,i)
            return z
        if dtyp=='XX':
            i,j=labels
            if rt=='x' and ct=='x': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            return z
        if dtyp=='W':
            if rt=='s' and ri=='0' and ct=='s' and ci=='w': return PExpr.one()
            if rt=='s' and ri=='w' and ct=='s' and ci=='0': return PExpr.one()
            if rt=='s' and ri=='w' and ct=='s' and ci=='w': return PExpr.one()
            return z
        if dtyp=='WX':
            i=labels[0]
            if rt=='x' and ct=='s' and ci=='w': return Delta(ri,i)
            if rt=='s' and ri=='w' and ct=='x': return Delta(ci,i)
            if ireg:
                if rt=='s' and ri=='0' and ct=='wx': return Delta(ci,i)
                if rt=='wx' and ct=='s' and ci=='0': return Delta(ri,i)
                if rt=='s' and ri=='w' and ct=='wx': return Delta(ci,i)
                if rt=='wx' and ct=='s' and ci=='w': return Delta(ri,i)
            return z
        if dtyp=='X2W':
            if not ireg: return z
            i,j=labels
            s=Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            if rt=='x' and ct=='wx': return s
            if rt=='wx' and ct=='x': return s
            if rt=='wx' and ct=='wx': return s
            return z
        return z
    return get
# Guarded replacement for the X2W block (the previous version attempted Delta on scalar labels).
def direction_dA(ireg,dtyp,labels):
    def get(row,col,pool=None):
        rt,ri=row; ct,ci=col; z=PExpr.zero()
        if dtyp=='X':
            i=labels[0]
            if rt=='s' and ri=='0' and ct=='x': return Delta(ci,i)
            if rt=='x' and ct=='s' and ci=='0': return Delta(ri,i)
        elif dtyp=='XX':
            i,j=labels
            if rt=='x' and ct=='x': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
        elif dtyp=='W':
            if rt=='s' and ri=='0' and ct=='s' and ci=='w': return PExpr.one()
            if rt=='s' and ri=='w' and ct=='s' and ci=='0': return PExpr.one()
            if rt=='s' and ri=='w' and ct=='s' and ci=='w': return PExpr.one()
        elif dtyp=='WX':
            i=labels[0]
            if rt=='x' and ct=='s' and ci=='w': return Delta(ri,i)
            if rt=='s' and ri=='w' and ct=='x': return Delta(ci,i)
            if ireg:
                if rt=='s' and ri=='0' and ct=='wx': return Delta(ci,i)
                if rt=='wx' and ct=='s' and ci=='0': return Delta(ri,i)
                if rt=='s' and ri=='w' and ct=='wx': return Delta(ci,i)
                if rt=='wx' and ct=='s' and ci=='w': return Delta(ri,i)
        elif dtyp=='X2W' and ireg:
            i,j=labels
            if rt=='x' and ct=='wx': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            if rt=='wx' and ct=='x': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
            if rt=='wx' and ct=='wx': return Delta(ri,i)*Delta(ci,j)+Delta(ri,j)*Delta(ci,i)
        return z
    return get

def _term_graph(factors):
    import networkx as nx
    G=nx.Graph()
    for fi,f in enumerate(factors):
        fn=f'F{fi}'; G.add_node(fn,bip='F',name=f.name)
        for s,i in enumerate(f.inds):
            inn=f'I{i}'
            if inn not in G: G.add_node(inn,bip='I',name='I')
            # Tensor symmetries make slot positions irrelevant for the symmetric
            # primitive tensors; retain positions only for genuinely ordered atoms.
            edge_slot=0 if f.name in SYMMETRIC else s
            G.add_edge(fn,inn,slot=edge_slot)
    return G

def coord_branches(typ,pool):
    ar=ARITY[typ]
    if ar==0: return [(typ,(),sp.Rational(1))]
    if typ in SYMPAIR:
        i=pool.fresh(); j=pool.fresh()
        # Ordered-pair representation: the derivative direction is E_ij+E_ji
        # and carries weight 1/2.  The two ordered orientations together reproduce
        # each off-diagonal symmetric coordinate once, while i=j gives (1/2)(2E_ii)=E_ii.
        return [(typ,(i,j),sp.Rational(1,2))]
    return [(typ,(pool.fresh(),),sp.Rational(1))]

def assert_einstein_incidence(facs):
    """Audit that every formal index in a closed monomial occurs exactly twice."""
    counts=Counter()
    for f in facs:
        counts.update(f.inds)
    bad={i:n for i,n in counts.items() if n != 2}
    if bad:
        raise AssertionError(f"invalid Einstein incidence {bad}: {facs}")

def closed_reduce(e:PExpr):
    """Reduce a closed scalar indexed expression. All indices are summed, so
    Delta[a,b] may always substitute b->a; Delta[a,a]=p. Then K Sigma and
    resulting deltas are contracted repeatedly. No free-coordinate indices are
    present when this routine is called."""
    out=defaultdict(lambda:sp.Integer(0))
    for fs,c in e.terms.items():
        assert_einstein_incidence(fs)
        facs=list(fs); coeff=sp.sympify(c); changed=True
        while changed:
            changed=False
            # Deltas first: closed scalar means unconditional identification.
            for pos,f in enumerate(facs):
                if f.name!='Delta': continue
                a,b=f.inds
                facs.pop(pos)
                if a==b:
                    coeff*=sp.Symbol('p')
                else:
                    facs=[Factor(h.name,tuple(a if x==b else x for x in h.inds)) for h in facs]
                changed=True; break
            if changed: continue
            counts=factor_counts(facs); found=False
            for i,f in enumerate(facs):
                if f.name!='K': continue
                for j,g in enumerate(facs):
                    if i==j or g.name!='Sigma': continue
                    for m in set(f.inds)&set(g.inds):
                        # m is necessarily summed in a closed expression.
                        a=f.inds[0] if f.inds[1]==m else f.inds[1]
                        b=g.inds[0] if g.inds[1]==m else g.inds[1]
                        facs=[h for k,h in enumerate(facs) if k not in (i,j)]
                        facs.append(Factor('Delta',(a,b))); found=True; changed=True; break
                    if found: break
                if found: break
        nf=[]
        for f in facs:
            inds=tuple(sorted(f.inds)) if f.name in SYMMETRIC else f.inds
            nf.append(Factor(f.name,inds))
        out[tuple(sorted(nf,key=lambda f:(f.name,f.inds)))]+=sp.expand(coeff)
    return PExpr(out)


# ===== Certificate driver (embedded from validation/general_p_symbolic_certificate.py) =====

"""Direct general-p indexed-sum certificate.

The proof object is the literal Hall expansion in formal feature indices.  p is
never instantiated.  All coordinate-index sums are generated mechanically;
XX/X2W symmetric coordinates are represented by their exact 1/2 ordered-sum
form.  The certificate subtracts the claimed invariant expression and requires
an exact symbolic zero remainder after tensor/dummy-index canonicalization.
"""
import time, json

ROWS_TARGET = {
    1: lambda: None,
}

def row_target(r):
    pool=IndexPool(); i,j,k,l,a,b,c,d=pool.many(8)
    G=PExpr.atom('gamma',i)*PExpr.atom('Sigma',i,j)*PExpr.atom('gamma',j)
    F=PExpr.atom('gamma',i)*PExpr.atom('gamma',j)*PExpr.atom('T4',i,j,k,l)*PExpr.atom('K',k,l)
    qg=PExpr.atom('gamma',i)*PExpr.atom('T3',i,k,l)*PExpr.atom('K',k,l)
    Zg=PExpr.atom('gamma',i)*PExpr.atom('Z',i,k,l)*PExpr.atom('K',k,l)
    Q=qg * (PExpr.atom('gamma',a)*PExpr.atom('T3',a,b,c)*PExpr.atom('K',b,c))
    R=PExpr.atom('gamma',i)*PExpr.atom('gamma',j)*PExpr.atom('T3',i,a,b)*PExpr.atom('K',a,c)*PExpr.atom('K',b,d)*PExpr.atom('T3',j,c,d)
    AP=PExpr.atom('gamma',i)*PExpr.atom('T3',i,j,k)*PExpr.atom('K',j,a)*PExpr.atom('P',a,b)*PExpr.atom('K',b,k)
    BP=qg*(PExpr.atom('K',c,d)*PExpr.atom('P',d,c))
    CP=PExpr.atom('gamma',i)*PExpr.atom('P',i,a)*PExpr.atom('K',a,j)*PExpr.atom('T3',j,k,l)*PExpr.atom('K',k,l)
    UM=PExpr.atom('K',i,j)*PExpr.atom('M',j,k)*PExpr.atom('K',k,l)*PExpr.atom('M',l,i)
    UN=PExpr.atom('T3',i,j,k)*PExpr.atom('K',j,k)*PExpr.atom('K',i,l)*PExpr.atom('N',l)
    Z=Zg
    if r==1: return 2*G+2*F+8*Z
    if r==2: return -3*G-F-Q-R-8*Z-8*AP-8*BP
    if r==3:
        p=sp.Symbol('p')
        return -(2*p+6)*G-2*R-8*AP+8*CP+8*UM+8*UN
    raise ValueError(r)

def row_hall(ireg,r,verbose=True):
    c=CachedSymbolics(ireg)
    types=COORD_TYPES_I if ireg else COORD_TYPES_R
    if r==1:
        ats=[t for t in types if c.get_a(t).terms]
        hp=[(a,b) for a in types for b in types if c.get_H(a,b).terms]
        total=PExpr.zero(); count=0
        for t1 in ats:
          for t2,t3 in hp:
            pool=IndexPool()
            acc=PExpr.zero()
            for insts,w in coord_combinations([t1,t2,t3],pool):
              (A,LA),(B,LB),(C,LC)=insts
              acc += (a_inst(c,A,LA,pool)*H_inst(c,B,LB,C,LC,pool)*cum3_fast(c,A,LA,B,LB,C,LC,pool)).scale(w)
            total=canonicalize(closed_reduce(total+closed_reduce(acc))); count+=1
            if verbose and count%10==0: print(' T1 combo',count,'/',len(ats)*len(hp),'terms',total.num_terms(),flush=True)
        return canonicalize(closed_reduce(total)),c
    if r==2:
        hp=[(a,b) for a in types for b in types if c.get_H(a,b).terms]
        total=PExpr.zero(); count=0
        for t1,t2 in hp:
          h1p=c.get_H(t1,t2)
          for t3,t4 in hp:
            h2p=c.get_H(t3,t4)
            pool=IndexPool(); acc=PExpr.zero()
            for insts,w in coord_combinations([t1,t2,t3,t4],pool):
              (A,LA),(B,LB),(C,LC),(D,LD)=insts
              h1=H_inst(c,A,LA,B,LB,pool); h2=H_inst(c,C,LC,D,LD,pool)
              wick=(cov_fast(c,A,LA,C,LC,pool)*cov_fast(c,B,LB,D,LD,pool)
                   +cov_fast(c,A,LA,D,LD,pool)*cov_fast(c,B,LB,C,LC,pool)
                   +cov_fast(c,A,LA,B,LB,pool)*cov_fast(c,C,LC,D,LD,pool))
              acc += h1*h2*wick.scale(w*sp.Rational(1,4))
            total=canonicalize(closed_reduce(total+closed_reduce(acc))); count+=1
            if verbose and count%20==0: print(' T2 combo',count,'/',len(hp)**2,'terms',total.num_terms(),flush=True)
        return canonicalize(closed_reduce(total)),c
    if r==3:
        ats=[t for t in types if c.get_a(t).terms]
        a3pats=[(a,b,d) for a in types for b in types for d in types if c.get_A3(a,b,d).terms]
        total=PExpr.zero(); count=0
        for t1 in ats:
          for t2,t3,t4 in a3pats:
            pool=IndexPool(); acc=PExpr.zero()
            for insts,w in coord_combinations([t1,t2,t3,t4],pool):
              (A,LA),(B,LB),(C,LC),(D,LD)=insts
              av=a_inst(c,A,LA,pool); a3=A3_inst(c,B,LB,C,LC,D,LD,pool)
              wick=(cov_fast(c,A,LA,B,LB,pool)*cov_fast(c,C,LC,D,LD,pool)
                   +cov_fast(c,A,LA,C,LC,pool)*cov_fast(c,B,LB,D,LD,pool)
                   +cov_fast(c,A,LA,D,LD,pool)*cov_fast(c,B,LB,C,LC,pool))
              acc += av*a3*wick.scale(w*sp.Rational(1,3))
            total=canonicalize(closed_reduce(total+closed_reduce(acc))); count+=1
            if verbose and count%20==0: print(' T3 combo',count,'/',len(ats)*len(a3pats),'terms',total.num_terms(),flush=True)
        return canonicalize(closed_reduce(total)),c


def run():
    results={}
    for r in (1,2,3):
      t=time.time(); print('=== row',r,'===',flush=True)
      R,cR=row_hall(False,r); print('REG terms',R.num_terms(), 'sec',time.time()-t,flush=True)
      t=time.time(); I,cI=row_hall(True,r); print('IREG terms',I.num_terms(), 'sec',time.time()-t,flush=True)
      D=canonicalize(closed_reduce(I-R)); T=canonicalize(closed_reduce(row_target(r))); rem=canonicalize(closed_reduce(D-T))
      results[r]={'reg_terms':R.num_terms(),'ireg_terms':I.num_terms(),'diff_terms':D.num_terms(),'target_terms':T.num_terms(),'remainder_terms':rem.num_terms()}
      print('DIFF terms',D.num_terms(),'TARGET terms',T.num_terms(),'REMAINDER terms',rem.num_terms(),flush=True)
      if rem.terms:
        print('REMAINDER',rem.terms,flush=True)
        raise AssertionError(f'row {r} remainder nonzero')
    print('PASS',json.dumps(results,indent=2))
    return results

if __name__=='__main__': run()
