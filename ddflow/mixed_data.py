import hashlib
import os
import pickle
from datetime import datetime
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import scipy
import torch
from PIL import Image
from rich.progress import track
from sklearn.datasets import make_sparse_spd_matrix
from survae.data.transforms import Quantize
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

BIOMARKER_PATH = "[FILL IN PATH TO EXTRACTED UKB BIOMARKER CSV]"
PHENO_PATH = "[FILL IN PATH TO UKB PHENOTYPE CSV]"


def get_pareto_rho_dataset(
    dset,
    N,
    N_valid,
    incl_inds=True,
    bs=128,
    alpha=0.5,
    rho_min=0.5,
    rho_max=0.99,
    block_size_min=1,
    block_size_max=1000,
    num_workers=0,
    **kwargs,
):
    print("ignoring parameters: ", kwargs)
    print(rho_min, rho_max)
    dset_fn = dict(
        sine=to_sine,
        ccube=to_crescent_cube,
        cres=to_crescent,
        abs=to_abs,
        sign=to_sign,
    )[dset]

    block_sizes = []
    while np.sum(block_sizes) < N:
        block_size = block_size_min + np.round(np.random.pareto(alpha))
        block_size = np.clip(
            block_size, None, min(block_size_max, N - np.sum(block_sizes))
        )
        block_sizes.append(int(block_size))

    rho_min, rho_max = sorted((rho_min, rho_max))
    rhos = list(np.random.uniform(low=rho_min, high=rho_max, size=len(block_sizes)))
    covs = [
        (1 - rho) * torch.eye(block_size) + rho
        for rho, block_size in zip(rhos, block_sizes)
    ]
    cov = scipy.linalg.block_diag(*covs)
    N = len(cov)
    X = []
    for cov_block in covs:
        ni = len(cov_block)
        block_sample = np.random.multivariate_normal(
            mean=np.zeros(ni), cov=cov_block, size=2
        ).T
        X.append(block_sample)
    X = np.concatenate(X)
    X = torch.from_numpy(X).float()
    xt = dset_fn(X)
    xv = dset_fn(torch.randn(N_valid, 2))
    xtt = dset_fn(torch.randn(N_valid, 2))

    if incl_inds:
        xt = TensorDataset(
            xt,
            torch.arange(len(xt)),
        )

    tl = DataLoader(xt, batch_size=bs, shuffle=True, num_workers=num_workers)
    vl = DataLoader(xv, batch_size=bs, shuffle=False, num_workers=num_workers)
    ttl = DataLoader(xtt, batch_size=bs, shuffle=False, num_workers=num_workers)
    ind_blocks = []
    counter = 0
    for block_size in block_sizes:
        ind_blocks.append([])
        for j in range(block_size):
            ind_blocks[-1].append(counter)
            counter += 1
    return tl, vl, ttl, dict(ind_blocks=ind_blocks, rhos=rhos, D=2, data_type="2d")


def get_cov_dataset(
    dset,
    N,
    N_valid,
    incl_inds=True,
    lam=0.5,
    chol_sparsity=0.0,
    chol_min=0.5,
    chol_max=0.99,
    extra_seed=None,
    sparsify=0,
    bs=128,
    **kwargs,
):
    """extra seed not necessary if pl.seed_everything is set"""
    print("ignoring parameters: ", kwargs)
    dset_fn = dict(
        sine=to_sine,
        ccube=to_crescent_cube,
        cres=to_crescent,
        abs=to_abs,
        sign=to_sign,
    )[dset]

    cov = make_sparse_spd_matrix(
        dim=N,
        alpha=chol_sparsity,
        smallest_coef=chol_min,
        largest_coef=chol_max,
        random_state=extra_seed,
        norm_diag=True,
    )
    if sparsify > 0:
        cov[np.abs(cov) < sparsify] = 0

    final_cov = lam * np.eye(N) + (1 - lam) * cov
    X = np.random.multivariate_normal(mean=np.zeros(N), cov=final_cov, size=2).T
    X = torch.from_numpy(X).float()
    xt = dset_fn(X)
    if incl_inds:
        xt = TensorDataset(
            xt,
            torch.arange(len(xt)),
        )
    tl = DataLoader(xt, batch_size=bs, shuffle=True)

    xv = dset_fn(torch.randn(N_valid, 2))
    vl = DataLoader(xv, batch_size=bs, shuffle=False)
    xtt = dset_fn(torch.randn(N_valid, 2))
    ttl = DataLoader(xtt, batch_size=bs, shuffle=False)

    return (
        tl,
        vl,
        ttl,
        dict(cov=cov, spectral=None, lam=lam, D=2, data_type="2d"),
    )


def to_sign(X):
    x1 = X[:, 0]
    x2_mean = torch.sign(x1) + x1
    x2_var = torch.exp(-3 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_abs(X):
    x1 = X[:, 0]
    x2_mean = torch.abs(x1) - 1
    x2_var = torch.exp(-3 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_crescent(X):
    x1 = X[:, 0]
    x2_mean = 0.5 * x1**2 - 1
    x2_var = torch.exp(-2 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_sine(X):
    x1 = X[:, 0]
    x2_mean = torch.sin(5 * x1)
    x2_var = torch.exp(-2 * torch.ones(x1.shape))
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x1, x2)).t()
    return x


def to_crescent_cube(X):
    x1 = X[:, 0]
    x2_mean = 0.2 * x1**3
    x2_var = torch.ones(x1.shape)
    x2 = x2_mean + x2_var**0.5 * X[:, 1]
    x = torch.stack((x2, x1)).t()
    return x


def get_torch_generator():
    g = torch.Generator()
    global_seed = torch.initial_seed() % 2**32
    g.manual_seed(global_seed)
    return g


def get_ukb_biomarker_data(incl_inds=True, bs=64, nan_threshold=0.95, seed=42):
    xt = UKBBiomarker(
        split="train", incl_inds=incl_inds, nan_threshold=nan_threshold, seed=seed
    )
    xv = UKBBiomarker(
        split="val", incl_inds=False, nan_threshold=nan_threshold, seed=seed
    )
    xtt = UKBBiomarker(
        split="test", incl_inds=False, nan_threshold=nan_threshold, seed=seed
    )

    tl = DataLoader(xt, batch_size=bs, shuffle=True, generator=get_torch_generator())
    vl = DataLoader(xv, batch_size=bs, shuffle=False)
    ttl = DataLoader(xtt, batch_size=bs, shuffle=False)

    spectral = xt.get_rm_spectral_decomposition()
    val_spectral = xv.get_rm_spectral_decomposition()
    test_spectral = xtt.get_rm_spectral_decomposition()

    D = xtt[0].shape[0]
    return (
        tl,
        vl,
        ttl,
        dict(
            lam=None,
            spectral=spectral,
            cov=None,
            D=D,
            data_type="mv",
            num_cols=5,
            val_cov=None,
            val_spectral=val_spectral,
            test_spectral=test_spectral,
            test_cov=None,
        ),
    )


def get_stock_pair_data(
    incl_inds=True,
    bs=64,
    start_year=2000,
    end_year=2012,
    ticks_or_pairs=["v-ma", "msft-aapl"],
    equi_or_rm="equi",
    seed=None,
):
    print(f"ignoring arg seed = {seed}")
    if "-" in ticks_or_pairs[0]:
        pairs = ticks_or_pairs
        ticks = None
    else:
        pairs = None
        ticks = ticks_or_pairs
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 1, 1)
    if equi_or_rm != "equi":
        raise NotImplementedError(equi_or_rm)
    xt = StockPairData(
        split="train",
        incl_inds=incl_inds,
        min_day=start,
        max_day=end,
        pairs=pairs,
        ticks=ticks,
    )
    xv = StockPairData(
        split="val",
        incl_inds=False,
        min_day=start,
        max_day=end,
        pairs=pairs,
        ticks=ticks,
    )
    xtt = StockPairData(
        split="test",
        incl_inds=False,
        min_day=start,
        max_day=end,
        pairs=pairs,
        ticks=ticks,
    )
    tl = DataLoader(xt, batch_size=bs, shuffle=True, generator=get_torch_generator())
    vl = DataLoader(xv, batch_size=bs, shuffle=False)
    ttl = DataLoader(xtt, batch_size=bs, shuffle=False)
    ind_blocks = xt.get_ind_blocks()
    D = 2
    return (
        tl,
        vl,
        ttl,
        dict(ind_blocks=ind_blocks, rhos=None, D=D, data_type="mv", num_cols=2),
    )


def get_adni_data(
    size=64,
    bs=32,
    val_bs=32,
    incl_inds=True,
    num_bits=8,
    equi_or_rm="equi",
    split_seed=42,
    num_workers=8,
):
    if size != 64:
        raise ValueError()
    tfm = []
    xt = ADNIMRI(
        split_seed=split_seed,
        split="train",
        pil_transforms=tfm,
        incl_inds=incl_inds,
        single_img_per_indiv=False,
        num_bits=num_bits,
    )
    xv1 = ADNIMRI(
        split_seed=split_seed,
        split="valid",
        pil_transforms=tfm,
        incl_inds=False,
        single_img_per_indiv=False,
        num_bits=num_bits,
    )
    xv2 = ADNIMRI(
        split_seed=split_seed,
        split="valid",
        pil_transforms=tfm,
        incl_inds=False,
        single_img_per_indiv=True,
        num_bits=num_bits,
    )
    xtt1 = ADNIMRI(
        split_seed=split_seed,
        split="test",
        pil_transforms=tfm,
        incl_inds=False,
        single_img_per_indiv=False,
        num_bits=num_bits,
    )
    xtt2 = ADNIMRI(
        split_seed=split_seed,
        split="test",
        pil_transforms=tfm,
        incl_inds=False,
        single_img_per_indiv=True,
        num_bits=num_bits,
    )

    tl = DataLoader(
        xt, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    vl_1 = DataLoader(
        xv1, batch_size=val_bs, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    vl_2 = DataLoader(
        xv2, batch_size=val_bs, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    ttl_1 = DataLoader(
        xtt1, batch_size=val_bs, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    ttl_2 = DataLoader(
        xtt2, batch_size=val_bs, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    if equi_or_rm == "rm":
        raise NotImplementedError(equi_or_rm)
    elif equi_or_rm == "equi":
        ind_blocks = xt.get_ind_blocks()
        return (
            tl,
            [vl_1, vl_2],
            [ttl_1, ttl_2],
            dict(
                ind_blocks=ind_blocks,
                rhos=None,
                data_type="img",
                D=(size**2),
                img_shape=(1, size, size),
            ),
        )
    else:
        raise ValueError(equi_or_rm)


class UKBBiomarker(Dataset):
    def __init__(
        self,
        biomarker_path=BIOMARKER_PATH,
        pheno_path=PHENO_PATH,
        split="train",
        train_pct=0.7,
        val_pct=0.15,
        incl_inds=True,
        seed=42,
        root="data_dir/ukb/",
        use_cached=True,
        nan_threshold=0.95,  # 0.95 -> ~3.2k; 0.9: 40k; 0.5: 240k
    ):
        self.incl_inds = incl_inds
        self.nan_threshold = nan_threshold

        cached_fn = join(root, f"ukbbiomarker_cache_{nan_threshold:.2f}_{seed}.pkl")
        if use_cached and os.path.isfile(cached_fn):
            (train_data, val_data, test_data), (
                train_gen,
                val_gen,
                test_gen,
            ) = pickle.load(open(cached_fn, "rb"))
        else:
            df = pd.read_csv(biomarker_path, index_col=0)
            df = df[[col for col in df.columns if col.endswith("0.0")]]
            df = df.loc[:, df.isna().mean() < nan_threshold].dropna()
            df = (df - df.mean()) / df.std()

            sniff = pd.read_csv(pheno_path, nrows=2)
            gen_cols = [
                col for col in sniff.columns[1:] if int(col.split("-")[0]) == 22009
            ]
            gen_pcs = pd.read_csv(
                pheno_path, index_col=0, usecols=["eid"] + gen_cols
            ).dropna()

            index = np.intersect1d(df.index, gen_pcs.index)
            df = df.loc[index]
            gen_pcs = gen_pcs.loc[index]

            rng = np.random.RandomState(seed)
            n = len(df)
            ind = rng.permutation(n)
            ntrain = int(np.round(train_pct * n))
            nval = int(np.round(val_pct * n))
            train_ind = ind[:ntrain]
            val_ind = ind[ntrain : (ntrain + nval)]
            test_ind = ind[(ntrain + nval) :]

            train_data = df.iloc[train_ind]
            val_data = df.iloc[val_ind]
            test_data = df.iloc[test_ind]
            train_gen = gen_pcs.iloc[train_ind]
            val_gen = gen_pcs.iloc[val_ind]
            test_gen = gen_pcs.iloc[test_ind]

            pickle.dump(
                [[train_data, val_data, test_data], [train_gen, val_gen, test_gen]],
                open(cached_fn, "wb"),
            )

        if split == "train":
            self.data = train_data
            self.gen = train_gen
        elif split in ["val", "valid"]:
            self.data = val_data
            self.gen = val_gen
        elif split == "test":
            self.data = test_data
            self.gen = test_gen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data.iloc[idx].values.astype(np.float32)
        obj = torch.from_numpy(obj).float()
        if self.incl_inds:
            return obj, idx
        else:
            return obj

    def get_relationship_matrix(self):
        return np.corrcoef(self.gen)

    def get_rm_spectral_decomposition(self, use_cached=True):
        cached_fn = f"data_dir/ukb/ukb_biomarkers_spectral_{self.nan_threshold:.2}.pkl"
        if use_cached and os.path.isfile(cached_fn):
            eigenvals, Q = pickle.load(open(cached_fn, "rb"))
        else:
            rm = self.get_relationship_matrix()
            eigh = torch.linalg.eigh(torch.from_numpy(rm).float())
            eigenvals, Q = eigh.eigenvalues, eigh.eigenvectors
            pickle.dump([eigenvals, Q], open(cached_fn, "wb"))
        return eigenvals, Q


class FWRotatedData(Dataset):
    @torch.no_grad()
    def __init__(self, dset, flow, device="cpu"):
        flow.to(device)
        latent = []
        ldjs = []
        for item in tqdm(dset):
            x, _ = item
            z, ldj = flow.data2noise(x[None].to(device), with_ldj=True)

            latent.append(z.cpu())
            ldjs.append(ldj.cpu())
        self.z = flow.base_dist.Q.T @ (
            torch.cat(latent).to(device) - flow.base_dist.loc
        )
        self.ldjs = torch.cat(ldjs)
        flow.cpu()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx]


class StockPairData(Dataset):
    def __init__(
        self,
        split="train",
        train_pct=0.7,
        val_pct=0.15,
        root="data_dir/stocks",
        incl_inds=True,
        min_day=datetime(2005, 1, 1),
        max_day=datetime(2015, 1, 1),
        use_cached=True,
        pairs=["fb-googl", "ibm-ge"],
        ticks=None,
    ):
        assert (pairs is None) or (ticks is None), "can't specify both ticks & pairs"
        assert not (
            (pairs is None) and (ticks is None)
        ), "need to specify either ticks or pairs"
        if not ticks is None:
            pairs = self._get_pairs(ticks)
        self.pairs = pairs
        self.root = root
        self.split = split
        self.incl_inds = incl_inds
        self.min_day = min_day
        self.max_day = max_day
        self.cols = "Open High Low Close Volume".split()
        cached_fn = join(
            root,
            hashlib.md5(
                f"stockpairdata_{min_day}_{max_day}_{'_'.join(sorted(pairs))}".encode(
                    "utf-8"
                )
            ).hexdigest()
            + ".pkl",
        )
        if use_cached and os.path.isfile(cached_fn):
            train, val, test = pickle.load(open(cached_fn, "rb"))
        else:
            data = self._preprocess_raw_data(root)
            data.index.name = "ts"
            data.sort_index(inplace=True)
            n = len(data)
            ntrain = int(np.round(train_pct * n))
            nval = int(np.round(val_pct * n))
            train = data.iloc[:ntrain].sort_values(["pair", "ts"])
            val = data.iloc[ntrain : (ntrain + nval)].sort_values(["pair", "ts"])
            test = data.iloc[(ntrain + nval) :].sort_values(["pair", "ts"])

            pickle.dump([train, val, test], open(cached_fn, "wb"))
        if split == "train":
            self.data = train
        elif split in ["val", "valid"]:
            self.data = val
        elif split == "test":
            self.data = test

        self.dep = self.data.pair

    def _get_pairs(self, ts):
        ts = sorted(ts)
        n = len(ts)
        return sorted([f"{ts[i]}-{ts[j]}" for i in range(n) for j in range(i + 1, n)])

    def get_stats(self):
        d = dict()
        tickers = set(ticker for pair in self.pairs for ticker in pair.split("-"))
        for ticker in tickers:
            d[ticker] = self.data.pair.apply(
                lambda x: x.startswith(ticker) or x.endswith(ticker)
            ).sum()
        return d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data.iloc[idx, :2].values.astype(np.float32)
        obj = torch.from_numpy(obj).float()
        if self.incl_inds:
            return obj, idx
        else:
            return obj

    def get_ind_blocks(self):
        dep = self.dep.values
        blocks = []
        last = None
        for i, s in enumerate(dep):
            if s == last:
                blocks[-1].append(i)
            else:
                blocks.append([i])
            last = s
        return blocks

    def get_relationship_matrix(self):
        ticker = self.dep.values[None]
        A_ids = (ticker == ticker.T).astype(float)
        days = self.data.index.values.reshape(-1, 1)
        A_days = np.exp(
            -self.time_decay_factor
            * (days - days.T).astype("timedelta64[D]").astype(int) ** 2
        )
        return 0.5 * A_ids + 0.5 * A_days

    def _preprocess_raw_data(self, root):
        tickers = sorted(set([x for pair in self.pairs for x in pair.split("-")]))
        ticker_fns = [join(root, "Stocks", f"{ticker}.us.txt") for ticker in tickers]

        data = []
        for ticker_fn in track(ticker_fns, description="reading ticker data..."):
            try:
                ticker_data = pd.read_csv(ticker_fn, index_col=0, parse_dates=True)
                ticker_data[self.cols] = np.log(ticker_data[self.cols]).diff()
                ticker_data["Ticker"] = ticker_fn.split("/")[-1].split(".")[0]
                rows = (ticker_data.index >= self.min_day) & (
                    ticker_data.index < self.max_day
                )
                ticker_data = ticker_data.loc[rows]
                data.append(ticker_data)
            except:
                print(f"couldnt read ticker {ticker_fn}")
        print("concat...")
        data = pd.concat(data)
        print("filtering...")
        data = data.loc[:, ["Close", "Ticker"]]
        data.dropna(inplace=True)

        all_pairs = []
        dates = data.index.unique()
        for date in dates:
            dd = data.loc[date]
            if isinstance(dd, pd.Series):
                continue
            date_pairs = []
            for pair in self.pairs:
                t1, t2 = pair.split("-")
                stock1 = dd[dd.Ticker == t1].Close
                stock2 = dd[dd.Ticker == t2].Close
                if len(stock1) > 0 and len(stock2) > 0:
                    date_pairs.append(
                        (stock1.values[0], stock2.values[0], f"{t1}-{t2}")
                    )
            all_pairs.append(
                pd.DataFrame(
                    date_pairs,
                    columns=["stock1", "stock2", "pair"],
                    index=[date] * len(date_pairs),
                )
            )
        data = pd.concat(all_pairs)

        print("finished preprocessing")
        return data


class ADNIMRI(Dataset):
    def __init__(
        self,
        root="data_dir/adni/",
        split_seed=42,
        train_pct=0.7,
        val_pct=0.15,
        split="train",
        pil_transforms=[],
        single_img_per_indiv=False,
        num_bits=8,
        incl_inds=True,
    ):
        self.incl_inds = incl_inds
        self.pil_transforms = pil_transforms
        self.tfm = Compose(pil_transforms + [ToTensor(), Quantize(num_bits)])

        indivs = np.array(os.listdir(root))
        N = len(indivs)
        rng = np.random.RandomState(split_seed)

        perm = rng.permutation(N)
        ntrain = int(np.round(train_pct * N))
        nval = int(np.round(val_pct * N))

        train_ind = perm[:ntrain]
        val_ind = perm[ntrain : (ntrain + nval)]
        test_ind = perm[(ntrain + nval) :]

        if split == "train":
            ind = train_ind
        elif split in ["val", "valid"]:
            ind = val_ind
        elif split == "test":
            ind = test_ind
        indivs = indivs[ind]
        paths = []
        iid = []
        i = 0
        for indiv in indivs:
            pths = sorted(glob(join(root, indiv, "*")))
            if single_img_per_indiv:
                pths = pths[:1]
            for p in pths:
                paths.append(p)
                iid.append(i)
            i += 1
        self.paths = paths
        self.iid = iid

    def __len__(self):
        return len(self.iid)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        img = Image.open(self.paths[index])
        if self.tfm is not None:
            img = self.tfm(img)
        if self.incl_inds:
            return img, index
        else:
            return img

    def get_ind_blocks(self):
        blocks = []
        counter = 0
        last = -1
        for iid in self.iid:
            if iid != last:
                blocks.append([counter])
            else:
                blocks[-1].append(counter)
            last = iid
            counter += 1
        return blocks
