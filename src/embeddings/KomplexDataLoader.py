import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

BUY = ('customer', 'buys', 'product')
REV_BUY = ('product', 'bought_by', 'customer')
BASKET = ('product', 'in_basket_of', 'customer')

# Aus den Retouren-Labels aller Events berechnet (auch des vorherzusagenden Kaufs), daher nie als Feature
LEAKY_COLUMNS = [
    'salesPerCustomer', 'returnsPerCustomer', 'customerReturnRate',
    'salesPerProduct', 'returnsPerProduct', 'productReturnRate',
]
LEAKY_PREFIXES = ('customerId_level_return_code_', 'variantID_level_return_code_')

LABEL_FEATURES = ['sales', 'returns', 'rate', 'hasHistory']

BASKET_FEATURES = ['basketSize', 'basketValue', 'basketAvgPrice', 'basketAvgDiscount']


def load_raw(data_dir, split='training'):
    customers = pd.read_pickle(f"{data_dir}/customer_nodes_{split}.p")
    products = pd.read_pickle(f"{data_dir}/product_nodes_{split}.p")
    events = pd.read_pickle(f"{data_dir}/event_table_{split}.p")
    return customers, products, events


class ExplicitRelationalDataLoader:
    def __init__(
        self,
        customer_data: pd.DataFrame,
        product_data: pd.DataFrame,
        event_data: pd.DataFrame,
        customer_features=('yearOfBirth', 'isMale', 'premier'),
        product_features=('avgGbpPrice', 'avgDiscountValue'),
        label_features=tuple(LABEL_FEATURES),
        product_type_prefix: str = "productType_",
        country_prefix: str = "Country_",
        brand_prefix: str = "Brand_",
        graph: str = "ausgelagert",
        article_node: bool = False,
        article_features: bool = False,
        basket: bool = False,
    ):
        self._checkFeatures(customer_features, product_features, label_features)
        if graph not in ("ausgelagert", "einfach"):
            raise ValueError(f"Unbekannter Graph '{graph}', erlaubt: 'ausgelagert', 'einfach'")
        if graph == "einfach" and article_node:
            raise ValueError("Der einfache Graph hat keine Zusatzknoten, Artikel-Information über article_features")

        self.df_customers = self._renameDuplicates(customer_data)
        self.df_products = self._renameDuplicates(product_data)
        self.df_events = event_data.copy()
        self.df_events['edge_id'] = np.arange(len(self.df_events))

        customer_map, product_map = self._reindex(self.df_customers, self.df_products)

        self.customer_return_ratio_cols = self._cols(self.df_customers, "customerId_level_return_code_")
        self.country_cols = self._cols(self.df_customers, country_prefix)
        self.product_type_cols = self._cols(self.df_products, product_type_prefix)
        self.brand_cols = self._cols(self.df_products, brand_prefix)

        self.data = HeteroData()
        self.data.customer_features = list(customer_features)
        self.data.product_features = list(product_features)
        self.data.label_features = list(label_features)
        self.data.group_nodes = {}
        self.data.group_features = {}

        self.data['customer'].x = self._createTensors(self.df_customers, list(customer_features))
        self.data['product'].x = self._createTensors(self.df_products, list(product_features))

        # Retourengrund-Anteile: nur für den Simulator, nie als Feature
        self.data['customer'].reason = self._createTensors(self.df_customers, self.customer_return_ratio_cols)

        if graph == "ausgelagert":
            self._add_category_node(
                df=self.df_products, src_type='product', dst_type='product_type',
                onehot_cols=self.product_type_cols, rel='has_type', rev_rel='type_of',
            )

            self._add_category_node(
                df=self.df_customers, src_type='customer', dst_type='country',
                onehot_cols=self.country_cols, rel='lives_in', rev_rel='home_of',
            )

            self._add_category_node(
                df=self.df_products, src_type='product', dst_type='brand',
                onehot_cols=self.brand_cols, rel='has_brand', rev_rel='brand_of',
            )
        else:
            self._add_onehot_features(self.df_customers, 'customer', self.country_cols)
            self._add_onehot_features(self.df_products, 'product', self.product_type_cols + self.brand_cols)

        if article_node:
            self._add_group_node(
                df=self.df_products, src_type='product', dst_type='article',
                key_col='hash(productID)', rel='variant_of', rev_rel='has_variant',
            )
        if article_features:
            self._add_group_features(df=self.df_products, member_type='product', group_name='article',
                                     key_col='hash(productID)')

        self.df_events['src_cust'] = self.df_events['hash(customerId)'].map(customer_map)
        self.df_events['dst_prod'] = self.df_events['hash(variantID)'].map(product_map)

        df_events_clean = self._filteringNaN(self.df_events, ['src_cust', 'dst_prod'])

        src_cust = df_events_clean['src_cust'].astype(np.int64).values
        dst_prod = df_events_clean['dst_prod'].astype(np.int64).values

        self.data[BUY].edge_index = self._edgeIndex(src_cust, dst_prod)
        self.data[BUY].y = torch.tensor(df_events_clean['isReturned'].values, dtype=torch.float)
        self.data[BUY].edge_id = torch.tensor(df_events_clean['edge_id'].values, dtype=torch.long)

        self.data[REV_BUY].edge_index = self._edgeIndex(dst_prod, src_cust)

        if basket:
            self._add_basket(src_cust, dst_prod)

    def _add_category_node(self, df, src_type, dst_type, onehot_cols, rel, rev_rel):
        values = df[onehot_cols].to_numpy()

        self.data[dst_type].x = torch.eye(len(onehot_cols), dtype=torch.float)

        active = values > 0
        has_category = active.sum(axis=1) >= 1

        n_missing = int((~has_category).sum())
        if n_missing > 0:
            print(f"Achtung: {n_missing} {src_type}-Knoten ohne {dst_type} (es wird keine Kante angelegt)")
        n_multi = int((active.sum(axis=1) > 1).sum())
        if n_multi > 0:
            print(f"Achtung: {n_multi} {src_type}-Knoten mit mehreren {dst_type}-Kategorien "
                  f"(nur die erste wird verwendet)")

        src = np.flatnonzero(has_category)
        dst = active[has_category].argmax(axis=1)

        self.data[src_type, rel, dst_type].edge_index = self._edgeIndex(src, dst)
        self.data[dst_type, rev_rel, src_type].edge_index = self._edgeIndex(dst, src)

    def _add_onehot_features(self, df, node_type, onehot_cols):
        values = torch.tensor(df[onehot_cols].to_numpy(dtype=np.float32) > 0, dtype=torch.float)
        self.data[node_type].x = torch.cat([self.data[node_type].x, values], dim=1)
        names = f"{node_type}_features"
        setattr(self.data, names, getattr(self.data, names) + list(onehot_cols))

    def _add_group_features(self, df, member_type, group_name, key_col):
        _, group = np.unique(df[key_col].to_numpy(), return_inverse=True)
        size = np.bincount(group)[group]
        self.data[member_type].x = torch.cat(
            [self.data[member_type].x, torch.tensor(size, dtype=torch.float).unsqueeze(1)], dim=1)
        names = f"{member_type}_features"
        setattr(self.data, names, getattr(self.data, names) + [f"{group_name}Variants"])
        self.data.group_features[member_type] = (group_name, torch.from_numpy(group.astype(np.int64)),
                                                 int(group.max()) + 1)

    def _add_basket(self, customers, products):
        n = self.data['customer'].num_nodes
        price = self.df_products['avgGbpPrice'].to_numpy()[products]
        discount = self.df_products['avgDiscountValue'].to_numpy()[products]

        size = np.bincount(customers, minlength=n).astype(float)
        value = np.bincount(customers, weights=price, minlength=n)
        discount_sum = np.bincount(customers, weights=discount, minlength=n)
        has_basket = size > 0
        avg_price = np.divide(value, size, out=np.zeros(n), where=has_basket)
        avg_discount = np.divide(discount_sum, size, out=np.zeros(n), where=has_basket)

        attrs = torch.tensor(np.stack([size, value, avg_price, avg_discount], axis=1), dtype=torch.float)
        self.data['customer'].x = torch.cat([self.data['customer'].x, attrs], dim=1)
        self.data.customer_features = self.data.customer_features + BASKET_FEATURES
        self.data[BASKET].edge_index = self._edgeIndex(products, customers)

    def _add_group_node(self, df, src_type, dst_type, key_col, rel, rev_rel):
        _, group = np.unique(df[key_col].to_numpy(), return_inverse=True)
        member = np.arange(len(df))

        self.data[dst_type].x = torch.tensor(np.bincount(group), dtype=torch.float).unsqueeze(1)
        self.data[src_type, rel, dst_type].edge_index = self._edgeIndex(member, group)
        self.data[dst_type, rev_rel, src_type].edge_index = self._edgeIndex(group, member)
        self.data.group_nodes[dst_type] = (src_type, rel, dst_type)

    @staticmethod
    def _checkFeatures(customer_features, product_features, label_features):
        for col in list(customer_features) + list(product_features):
            if col in LEAKY_COLUMNS or col.startswith(LEAKY_PREFIXES):
                raise ValueError(
                    f"'{col}' wurde aus den Retouren-Labels berechnet und darf kein Feature sein. "
                    f"Label-abhängige Features über label_features auswählen: {LABEL_FEATURES}"
                )
        unknown = set(label_features) - set(LABEL_FEATURES)
        if unknown:
            raise ValueError(f"Unbekannte label_features {sorted(unknown)}, erlaubt: {LABEL_FEATURES}")

    @staticmethod
    def _renameDuplicates(df: pd.DataFrame) -> pd.DataFrame:
        # betrifft '*_level_return_code_D', das in den Rohdaten doppelt vorkommt
        cols, seen = list(df.columns), set()
        for i, col in enumerate(cols):
            if col in seen:
                cols[i] = f"{col}_extra"
            seen.add(col)
        df = df.copy()
        df.columns = cols
        return df

    @staticmethod
    def _cols(df: pd.DataFrame, prefix: str) -> list:
        cols = [c for c in df.columns if str(c).startswith(prefix)]
        if not cols:
            raise ValueError(
                f"Keine Spalten mit Prefix '{prefix}' gefunden. Vorhandene Spalten: {list(df.columns)}"
            )
        return cols

    @staticmethod
    def _edgeIndex(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.stack([src, dst]).astype(np.int64))

    def _reindex(self, df_customers: pd.DataFrame, df_products: pd.DataFrame):
        if not df_customers['hash(customerId)'].is_unique:
            raise ValueError("hash(customerId) ist nicht eindeutig")
        if not df_products['hash(variantID)'].is_unique:
            raise ValueError("hash(variantID) ist nicht eindeutig")
        customer_map = {id_: i for i, id_ in enumerate(df_customers['hash(customerId)'])}
        product_map = {id_: i for i, id_ in enumerate(df_products['hash(variantID)'])}
        return customer_map, product_map

    def _createTensors(self, df: pd.DataFrame, features: list) -> torch.Tensor:
        return torch.tensor(df[features].values.copy(), dtype=torch.float)

    def _filteringNaN(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        for feature in features:
            missing = df[feature].isna().sum()
            if missing > 0:
                print(f"Achtung: {missing} ({(missing / df.shape[0]) * 100:.2f}%) Events ohne {feature}-Knoten")
        return df.dropna(subset=features).copy()

    @staticmethod
    def _scaleTensor(x: torch.Tensor, mean=None, std=None):
        if mean is None and std is None:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
        return ((x - mean) / std), mean, std
