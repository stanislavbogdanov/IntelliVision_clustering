import pandas as pd
import numpy as np
import math
import gc
import os
import psutil

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.optimize import linear_sum_assignment

from joblib import Parallel, delayed
from tqdm import tqdm
# from tqdm.notebook import tqdm

from typing import Literal

from sklearn.base import clone
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, FunctionTransformer
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, trustworthiness
import umap
import hdbscan  # Авторская реализация hdbscan, не вошедшая в sklearn

from tempfile import mkdtemp
from shutil import rmtree


# ------------------------------------
# Оптимальное количество компонент PCA
# ------------------------------------

def get_sample(X, sample_frac=0.3, dtype=np.float32, random_state=42, verbose=True):
    """
    Случайная выборка, принудительное преобразование в float32
    """
    n_rows = X.shape[0]
    sample_size = max(
        1,
        int(n_rows * np.clip(sample_frac, a_min=0.0, a_max=1.0))
    )
    rng = np.random.RandomState(random_state)
    rand_indices = rng.choice(n_rows, size=sample_size, replace=False)
    if X.dtype == dtype:
        X_sample = X[rand_indices]
    else:
        X_sample = X[rand_indices].astype(dtype)
    if verbose:
        print(f"Используется выборка: {n_rows} -> {sample_size} строк ({sample_size/n_rows:.1%})")
    return X_sample


def horn_parallel_analysis(
    X_data, 
    n_components_eval=100, 
    n_iterations=20, 
    quantile=0.95, 
    sample_frac=1.0, 
    random_state=42
):
    """
    Параллельный анализ Хорна на основе перестановок (Permutation Parallel Analysis).
    
    Параметры:
    - X_data: матрица признаков
    - n_components_eval: количество первых компонент для исследования
    - n_iterations: количество симуляций шума
    - quantile: порог отсечения (95-й процентиль шума)
    - sample_frac: размер подвыборки для ускорения расчета
    """
    np.random.seed(random_state)
    
    # 1. Берем подвыборку для быстрого расчета собственных чисел
    X_sample = get_sample(X_data, sample_frac=sample_frac, random_state=random_state, verbose=False)
        
    _, n_features = X_sample.shape
    n_components_eval = min(n_components_eval, n_features)
    
    # 2. Считаем PCA для РЕАЛЬНЫХ данных
    pca_real = PCA(n_components=n_components_eval, svd_solver='randomized', random_state=random_state)
    pca_real.fit(X_sample)
    real_eigenvalues = pca_real.explained_variance_
    
    # 3. Симуляция: перемешиваем каждую колонку независимо (сохраняем распределение, убираем связи)
    simulated_eigenvalues = np.zeros((n_iterations, n_components_eval))
    X_permuted = np.empty_like(X_sample)
    
    for b in range(n_iterations):
        for j in range(n_features):
            X_permuted[:, j] = np.random.permutation(X_sample[:, j])
            
        pca_sim = PCA(n_components=n_components_eval, svd_solver='randomized', random_state=random_state + b)
        pca_sim.fit(X_permuted)
        simulated_eigenvalues[b, :] = pca_sim.explained_variance_
        
    # 4. Порог отсечения: 95-й процентиль шума
    noise_threshold = np.quantile(simulated_eigenvalues, quantile, axis=0)
    
    # 5. Определение значимых компонент (сигнал > шум)
    significant_mask = real_eigenvalues > noise_threshold
    if np.any(significant_mask):
        k_optimal = np.where(significant_mask)[0][-1] + 1
    else:
        k_optimal = 1
        
    min_variance_ratio = np.sum(pca_real.explained_variance_ratio_[:k_optimal])
    
    return {
        "k_optimal": k_optimal,                     # Количество значимых компонент
        "min_pca_variance": min_variance_ratio,     # Доля объясненной дисперсии
        "real_eigenvalues": real_eigenvalues,       # Собственные числа
        "noise_threshold": noise_threshold          # Порог отсечения шума
    }


def get_trans_name(trans_obj):
    """Имя трансформера для отчета"""
    return "None" if trans_obj is None else trans_obj.__class__.__name__


def get_optimal_pca_components(
        X, 
        transformers, 
        n_components_eval=None, 
        n_iterations=20, 
        sample_frac=1.0, 
        n_jobs=None, 
        random_state=42
):
    """
    Находит минимально необходимое число компонент PCA/SVD методом Хорна, 
    перебирая варианты предварительной трансформации признаков.

    Параметры:
    - X: исходный датасет
    - transformers: Список или словарь sklearn-совместимых трансформеров
    - n_components_eval: количество первых компонент для исследования
    - sample_frac: доля строк исходного датасета, которая используется для оценки
    """
    def evaluate_distorsion(X_data, distorsion, n_components_eval, n_iterations, sample_frac, random_state):
        """Функция оценки оптимального числа компонент для одной комбинации параметров"""
        try:
            trans_name = get_trans_name(distorsion)
            prep_pipe = Pipeline([
                ("distorsion", clone(distorsion)),
                ("scaling", StandardScaler())
            ])
            X_trans = prep_pipe.fit_transform(X_data)
            horn = horn_parallel_analysis(
                X_trans, 
                n_components_eval=n_components_eval,
                n_iterations=n_iterations,
                sample_frac=sample_frac, 
                random_state=random_state
            )
            return horn['k_optimal']
        
        except Exception as e:
            print(f"Ошибка на {trans_name=}: {e}")
            return None

    if isinstance(transformers, dict):
        transformers_dict = transformers
    elif isinstance(transformers, list):
        transformers_dict = {get_trans_name(t):t for t in transformers}
    else:
        raise ValueError('get_optimal_pca_components: transformers должен быть списком или словарем.')
    
    # Если число компонент не указано, то уменьшить размерность минимум в 5 раз
    n_components_eval = n_components_eval if n_components_eval else int(X.shape[1]/5)

    # Если n_jobs не указан, то использовать однозадачный режим (максимальная совместимость)
    n_jobs = n_jobs if n_jobs else 1

    n_components = Parallel(n_jobs=min(n_jobs, len(transformers_dict)), backend="loky")(
        delayed(evaluate_distorsion)(
            X, trans_obj, 
            n_components_eval=n_components_eval, 
            n_iterations=n_iterations, 
            sample_frac=sample_frac, 
            random_state=random_state
        ) 
        for trans_obj in tqdm(transformers_dict.values(), desc="Трансформеры")
    )

    # joblib.Parallel сохраняет исходный порядок результатов в списке в соответствии с переданным итератором
    n_components_dict = {
        name: k 
        for name, k in zip(transformers_dict.keys(), n_components) 
        if k is not None
    }    

    return n_components_dict


# -------------------------------
# Метрики
# -------------------------------

def hopkins_statistic(X, sample_frac=0.2, n_jobs=None, random_state=42):
    """
    Статистика Хопкинса.
    - H ≈ 0.5: Данные случайны (нет структуры)
    - H → 1.0: Данные сильно кластеризованы (есть чёткие кластеры)
    - H → 0.0: Данные равномерно распределены (регулярная сетка)
    """

    # Если n_jobs не указан, то использовать однозадачный режим (максимальная совместимость)
    n_jobs = n_jobs if n_jobs else 1

    # Случайная выборка
    X_sample = get_sample(X, sample_frac=sample_frac, random_state=random_state, verbose=False)

    # Генерация равномерных случайных точек
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    X_uniform = np.random.uniform(low=min_val, high=max_val, size=X_sample.shape)

    # wdist: расстояния до ближайших соседей в реальных данных для случайных точек из данных
    nnw = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=n_jobs)
    nnw.fit(X)
    sample_dist, _ = nnw.kneighbors(X_sample)
    wdist = sample_dist[:, 1]  # Сосед №2, чтобы не взять ту же самую точку

    # udist: расстояния до ближайших соседей в реальных данных для равномерных точек
    nnu = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=n_jobs)
    nnu.fit(X)
    uniform_dist, _ = nnu.kneighbors(X_uniform)
    udist = uniform_dist[:, 0] 

    # Статистика Хопкинса 
    sum_w = np.sum(wdist)
    sum_u = np.sum(udist)
    H = sum_u / (sum_w + sum_u)
    return H


def cluster_stability(X_data, n_clusters, n_repeats=5, sample_frac=0.8, random_state=42):
    """
    Оценка стабильности кластеризации через бутстрап/подвыборки и ARI.
    Идея: Если найденные кластеры отражают реальные паттерны, а не случайный шум, то они устойчивы к изменению выборки.

    0.85 - 1.00	Идеальная	В данных существуют четкие, плотные, устойчивые кластеры. Границы воспроизводятся почти на 100%.
    0.70 - 0.85	Хорошая	    Кластерная структура выражена хорошо, но на границах есть небольшое количество пограничных объектов.
    0.50 - 0.70	Умеренная	Кластеры сильно пересекаются или имеют сложную, несферическую форму.
    < 0.50	Нестабильная	Кластеров нет (однородное распределение/шум), алгоритм на каждом запуске нарезает пространство случайным образом.        
    
    Параметры:
    - X_data: матрица признаков
    - n_clusters: число кластеров
    - n_repeats: количество экспериментов (бутстреп)
    - sample_frac: размер подвыборки (обычно 80%)
    """
    n_samples = X_data.shape[0]
    subsample_size = int(n_samples * sample_frac)
    
    ari_scores = []
    rng = np.random.RandomState(random_state)
    batch_size = min(2048, X_data.shape[0])
    
    for i in range(n_repeats):
        # Генерируем индексы для двух независимых подвыборок
        idx_1 = rng.choice(n_samples, size=subsample_size, replace=False)
        idx_2 = rng.choice(n_samples, size=subsample_size, replace=False)
        
        # Находим пересечение (общие объекты для обеих подвыборок)
        intersection_idx = np.intersect1d(idx_1, idx_2, assume_unique=True)
        
        # Обучаем две независимые модели на своих подвыборках
        kmeans_1 = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=batch_size, 
            random_state=rng.randint(0, 100000)
        ).fit(X_data[idx_1])
        
        kmeans_2 = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=batch_size, 
            random_state=rng.randint(0, 100000)
        ).fit(X_data[idx_2])
        
        # Предсказываем метки для ОБЩИХ объектов
        labels_1 = kmeans_1.predict(X_data[intersection_idx])
        labels_2 = kmeans_2.predict(X_data[intersection_idx])
        
        # Считаем согласованность разбиений (ARI)
        ari = adjusted_rand_score(labels_1, labels_2)
        ari_scores.append(ari)
        
    return {
        "stability_mean": np.mean(ari_scores), # Средняя стабильность
        "stability_std": np.std(ari_scores)    # Разброс стабильности
    }


# ---------------------------------
# Подбор числа компонентов PCA
# ---------------------------------

def evaluate_clustering(
    trans_name, 
    X_trans, 
    n_clusters, 
    pca_variance, 
    n_pca_components_actual, 
    hopkins_stat, 
    random_state=42
):
    """
    Кластеризация на n_clusters и оценка результата
    """
    try:
        batch_size = min(2048, X_trans.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
        labels = kmeans.fit_predict(X_trans)
        
        # Расчет метрик в ПРЕОБРАЗОВАННОМ пространстве
        ch = calinski_harabasz_score(X_trans, labels)
        db = davies_bouldin_score(X_trans, labels)

        # Доля наименьшего кластера
        min_cluster_pct = np.bincount(labels).min() / len(labels)

        # Стабильность кластеризации
        stab = cluster_stability(X_trans, n_clusters=n_clusters)

        return {
            "transformer": trans_name,
            "pca_variance": pca_variance,
            "n_pca_components_actual": n_pca_components_actual,
            "hopkins_stat": hopkins_stat,
            "n_clusters": n_clusters,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "stability": stab["stability_mean"],
            "stability_std": stab["stability_std"],
            "min_cluster_pct": min_cluster_pct
        }
    except Exception as e:
        print(f"Ошибка при кластеризации '{trans_name}', k={n_clusters}: {e}")
        # raise e
        return None


def clustering_measure(X, transformers, components, n_clusters, n_jobs=None, random_state=42):
    """
    Реализует пайплайн 
    1. "distorsion": предварительная трансформация признаков,
    2. StandardScaler,
    3. PCA с оптимальным количеством компонент, переданным в параметре n_components,
    4. кластеризация методом MiniBatchKMeans.
    
    Перебирает варианты distorsion и количества кластеров n_clusters, для каждого указывает
    - количество компонент, долю объясненной дисперсии в PCA и статистику Хопкинса,
    - индекс Калински-Харабаса, 
    - индекс Девиса-Болдина,
    - стабильность кластеризации,
    - долю наименьшего кластера.
    Возвращает результат в виде таблицы pandas Dataframe.

    Параметры:
    - X: исходный датасет.
    - transformers: Список или словарь sklearn-совместимых трансформеров.
    - components: Оптимальное количество компонент PCA - список или словарь, синхронизированный с transformers.
    - n_clusters: Список или итератор с вариантами количества кластеров.
    """

    # Подготовка переданных итераторов
    if isinstance(transformers, list):
        transformers_dict = {get_trans_name(t): t for t in transformers}
    else:
        transformers_dict = transformers

    if isinstance(components, list):
        components_dict = dict(zip(transformers_dict.keys(), components))
    else:
        components_dict = components

    # Если n_jobs не указан, то использовать однозадачный режим (максимальная совместимость)
    n_jobs = n_jobs if n_jobs else 1

    # Последовательный цикл по трансформерам: в каждый момент в памяти находится только 1 матрица
    all_results = []
    for trans_name, trans_obj in tqdm(transformers_dict.items(), desc="Трансформеры"):
        
        # 1. Предобработка, PCA и расчет статистики Хопкинса
        pipe_pca = Pipeline([
            ("distorsion", trans_obj),
            ("scaling", StandardScaler()),
            ("pca", PCA(n_components=components_dict[trans_name], svd_solver='randomized', random_state=random_state))
        ])
        
        # Трансформация и сжатие данных
        X_trans = pipe_pca.fit_transform(X)

        # Расчет статистики Хопкинса на преобразованных данных
        hstat = hopkins_statistic(X_trans, random_state=random_state)

        # Доля объясненной дисперсии
        pca_variance = float(np.sum(pipe_pca[-1].explained_variance_ratio_))

        # Количество компонент PCA
        n_pca_components_actual = X_trans.shape[1]

        # 2. Параллельная кластеризация по всем n_clusters
        # joblib с backend='loky' автоматически использует memory-mapping (memmap) на диск,
        # поэтому все воркеры читают X_trans из одного участка памяти/диска без копирования.
        tasks = [
            delayed(evaluate_clustering)(
                trans_name=trans_name,
                X_trans=X_trans,
                n_clusters=k,
                pca_variance=pca_variance,
                n_pca_components_actual=n_pca_components_actual,
                hopkins_stat=hstat,
                random_state=random_state
            )
            for k in n_clusters
        ]

        cachedir = mkdtemp() # Можно указать путь к быстрому SSD
        try:
            batch_results = Parallel(
                n_jobs=min(n_jobs, len(n_clusters)), 
                backend="loky",
                temp_folder=cachedir,  
                max_nbytes="100M"           # Все массивы > 100 МБ автоматически кэшируются через memmap
            )(tasks)

        finally:
            rmtree(cachedir, ignore_errors=True)  # Очистка диска от временного кэша

        all_results.extend([r for r in batch_results if r is not None])

        # 3. Принудительно освободить память перед переходом к следующему трансформеру
        del X_trans, pipe_pca
        gc.collect()

    return pd.DataFrame(all_results)


def umap_hdbscan_measure(
    X, 
    sample_frac=0.3, 
    umap_n_components=2, 
    umap_n_neighbors=300,  # Увеличено для макро-кластеров
    umap_metric='euclidean',
    trustworthiness_n_neighbors=30,
    cluster_selection_epsilon=0.0, 
    min_samples=10,
    ram_factor=0.8,
    n_jobs=None,
    random_state=42
):
    """
    Оценивает по метрикам trustworthiness и доли шума связку UMAP+HDBSCAN, перебирая
    - umap_n_components: int или list - размерность UMAP,
    - cluster_selection_epsilon: float или list - порог объединения кластеров (параметр HDBSCAN),
    - min_samples: int или list - количество соседей у ядерной точки (параметр HDBSCAN).

    Следующие параметры фиксируются:
    - umap_n_neighbors: int - по умолчанию 300 (регулирует охват глобальных структур)
    - umap_metric: метрика расстояния ('euclidean', 'cosine' и др.)
    - min_cluster_size: 2% данных
    - trustworthiness_n_neighbors: int - по умолчанию 30

    ram_factor: доля доступной памяти (физическая + swap), которая может быть отдана 
    для вычисления trustworthiness.

    Возвращает pandas DataFrame.
    """

    # Если n_jobs не указан, то использовать однозадачный режим (максимальная совместимость)
    n_jobs = n_jobs if n_jobs else 1

    # Случайная выборка, принудительное преобразование в float32
    X_sample = get_sample(X, sample_frac=sample_frac, random_state=random_state, verbose=True)
    minclustersize = max(1, int(X_sample.shape[0]*0.02))

    # Подготовка к семплированию X_sample, X_umap при нехватке памяти
    # Доступная память: физическая + swap
    available_ram = psutil.virtual_memory().available + psutil.swap_memory().free
    target_ram = available_ram * ram_factor
    arr_itemsize = X_sample.dtype.itemsize
    idx_itemsize = np.dtype(np.intp).itemsize  # 8 байт (np.argsort возвращает целочисленные индексы)
    # Оценка памяти на пару точек в матрицах N x N:
    # 1 матрица расстояний X_sample + 1 матрица расстояний X_umap + 1 матрица индексов argsort 
    bytes_per_point = arr_itemsize*2 + idx_itemsize
    # Максимально допустимое число точек для trustworthiness
    trustworthiness_max_n = int(np.sqrt(target_ram / bytes_per_point))
    if trustworthiness_max_n <= trustworthiness_n_neighbors:
        raise MemoryError(
            f"Недостаточно памяти: допустимо максимум N={trustworthiness_max_n}, "
            f"что меньше {trustworthiness_n_neighbors=}."
        )

    # Подвыборка строк для trustworthiness при необходимости
    if X_sample.shape[0] > trustworthiness_max_n:
        rng = np.random.RandomState(random_state)
        tw_indices = rng.choice(X_sample.shape[0], size=trustworthiness_max_n, replace=False)
        X_sample_sub = X_sample[tw_indices]
        reduct_factor = trustworthiness_max_n / X_sample.shape[0]
        print(f"Для trustworthiness выборка уменьшена: {X_sample.shape[0]} -> {trustworthiness_max_n} строк ({reduct_factor:.1%})")

    # Подготовка параметров для перебора
    umap_dim_values = np.atleast_1d(umap_n_components).tolist()
    eps_values = np.atleast_1d(cluster_selection_epsilon).tolist()
    min_samples_values = np.atleast_1d(min_samples).tolist()

    results = []
    for d in tqdm(umap_dim_values, desc='umap_dim'):
        # print(f'{d=}, {umap_n_neighbors=}, {umap_metric=}')
        reducer = umap.UMAP(
            n_components=d, 
            n_neighbors=umap_n_neighbors,    
            min_dist=0.0,       # Максимально уплотняет точки кластера (для кластеризации)
            metric=umap_metric, 
            n_jobs=n_jobs,
            random_state=42
        )
        X_umap = reducer.fit_transform(X_sample)

        # Оценка сохранения топологии
        if X_sample.shape[0] > trustworthiness_max_n:
            X_umap_sub = X_umap[tw_indices]
            trust = trustworthiness(X_sample_sub, X_umap_sub, n_neighbors=trustworthiness_n_neighbors)
        else:
            trust = trustworthiness(X_sample, X_umap, n_neighbors=trustworthiness_n_neighbors)

        for eps in eps_values:
            for minsmp in min_samples_values:
                # print(f'{minclustersize=}, {minsmp=}, {eps=}')
                # Кластеризация
                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=minclustersize, 
                    min_samples=minsmp,     # Консервативность оценки плотности
                    cluster_selection_epsilon=eps,
                    cluster_selection_method="eom",
                    core_dist_n_jobs=n_jobs
                )
                labels = hdb.fit_predict(X_umap)
                if sum(labels >= 0) == 0:
                    continue

                n_found_clusters = len(np.unique(labels[labels >= 0]))
                noise_ratio = np.mean(labels == -1)

                # Доля наименьшего кластера
                min_cluster_cnt = np.bincount(labels[labels >= 0]).min()
                min_cluster_pct = min_cluster_cnt / len(labels[labels >= 0])

                # Стабильность
                # stability = cluster_stability(X_umap, n_found_clusters)

                # DBCV (Density-Based Cluster Validation)
                # Математический аналог силуэта, созданный специально для плотностных методов. Он не использует центроиды, 
                # а рассчитывает плотность взаимной достижимости (mutual reachability distance) вдоль плотностных графов внутри 
                # кластеров и в межкластерных просветах. Диапазон от −1 (ужасно) до +1 (идеальное плотностное разделение).
                # if X_sample.shape[0] > trustworthiness_max_n:
                #     dbcv = hdbscan.validity.validity_index(X_umap_sub.astype(np.float64), labels[tw_indices])
                # else:
                #     dbcv = hdbscan.validity.validity_index(X_umap.astype(np.float64), labels)

                results.append({
                    "eps": eps,
                    "min_samples": minsmp,
                    "umap_dim": d,
                    "trustworthiness": trust,
                    "clusters_found": n_found_clusters,
                    "noise_pct": noise_ratio,
                    "min_cluster_pct": min_cluster_pct,
                    "min_cluster_cnt": min_cluster_cnt
                    # "stability_mean": stability['stability_mean'],
                    # "stability_std": stability['stability_std'],
                    # "DBCV": dbcv
                })
    df_result = pd.DataFrame(results).sort_values(['eps', 'min_samples', 'umap_dim'], ignore_index=True)
    return df_result


# ---------------------------------
# Поиск решения на множестве Парето
# ---------------------------------

def is_pareto_efficient(df, ch_col="calinski_harabasz", db_col="davies_bouldin", stab_col="stability"):
    """
    Поиск Парето-оптимальных решений для трех метрик:
    - CH: максимизация (>=)
    - DB: минимизация (<=)
    - Stability: максимизация (>=)
    
    Возвращает:
    - np.ndarray (bool): маска True/False для каждой строки DataFrame
    """
    ch = df[ch_col].to_numpy()
    db = df[db_col].to_numpy()
    stab = df[stab_col].to_numpy()
    
    n_points = len(df)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        
        # 1. Точка j не хуже точки i по всем трем критериям:
        as_good = (ch >= ch[i]) & (db <= db[i]) & (stab >= stab[i])
        
        # 2. И строго лучше хотя бы по одному критерию:
        strictly_better = (ch > ch[i]) | (db < db[i]) | (stab > stab[i])
        
        # Если нашлась хотя бы одна такая точка j, то точка i удаляется из множества Парето
        if np.any(as_good & strictly_better):
            is_pareto[i] = False
            
    return is_pareto


def pareto_decision(df_metrics, min_cluster_threshold=0.02):
    """
    Выбирает лучшие решения из датафрейма, который возвращает evaluate_pipeline.

    Параметры:
    - min_cluster_threshold: часть выборки, ниже которой не может быть минимальный кластер, 
    чтобы решение не было отнесено к вырожденным. По умолчанию 2%.
    """

    # 1. Фильтрация вырожденных решений (минимальный кластер >= 2% выборки)
    df_filtered = df_metrics[df_metrics["min_cluster_pct"] >= min_cluster_threshold].copy()

    # 2. MinMax нормализация метрик: приводим обе к шкале "больше - лучше" и масштабируем в диапазон [0,1]
    ch_min, ch_max = df_filtered["calinski_harabasz"].min(), df_filtered["calinski_harabasz"].max()
    db_min, db_max = df_filtered["davies_bouldin"].min(), df_filtered["davies_bouldin"].max()
    stab_min, stab_max = df_filtered["stability"].min(), df_filtered["stability"].max()

    df_filtered["ch_norm"] = (df_filtered["calinski_harabasz"] - ch_min) / (ch_max - ch_min + 1e-9)
    df_filtered["db_norm"] = (db_max - df_filtered["davies_bouldin"]) / (db_max - db_min + 1e-9) # Для DB инвертируем, т.к. меньше=лучше
    df_filtered["stab_norm"] = (df_filtered["stability"] - stab_min) / (stab_max - stab_min + 1e-9)

    # 3. Композитная метрика (для стабильности чуть больший вес) 
    df_filtered["composite_score"] = 0.3 * df_filtered["ch_norm"] + 0.3 * df_filtered["db_norm"] + 0.4 * df_filtered["stab_norm"]

    # 4. Поиск границы Парето (Pareto Front)
    df_filtered["is_pareto"] = is_pareto_efficient(df_filtered)

    # 5. Вывод ТОП решений
    pareto_leaders = df_filtered[df_filtered["is_pareto"]].sort_values(by="composite_score", ascending=False)
    pareto_leaders.drop(columns=[
        "ch_norm", "db_norm", "stab_norm", "is_pareto"
    ], inplace=True)
    return pareto_leaders


# -------------------------------
# Пайплайн для кластеризации
# -------------------------------

class PipeKMeansCluster(ClusterMixin, BaseEstimator):
    def __init__(
        self, pca_n_components, 
        n_clusters, 
        distorsion: Literal['RobustScaler', 'Yeo-Johnson', 'QuantileTransform'],
        distorsion_params=None,
        sample_frac=1.0, 
        random_state=42
    ):
        self.pca_n_components = pca_n_components
        self.n_clusters = n_clusters
        self.distorsion = distorsion
        self.distorsion_params = distorsion_params
        self.sample_frac = np.clip(sample_frac, a_min=0.0, a_max=1.0)
        self.random_state = random_state
        # rng = np.random.RandomState(random_state)

        self.X_sample = None
        self.rand_indices = None
        self.X_pca = None
        self.kmeans = None
        self.labels_ = None

        # Трансформация, стандартизация и уменьшение размерности
        pipeline = []
        if distorsion_params is None: distorsion_params = {}
        if distorsion == 'RobustScaler':
            pipeline.append(("distorsion", RobustScaler(**distorsion_params)))
        elif distorsion == 'Yeo-Johnson':
            pipeline.append(("distorsion", PowerTransformer(method="yeo-johnson", **distorsion_params)))
        elif distorsion == 'QuantileTransform':
            pipeline.append((
                "distorsion", 
                QuantileTransformer(
                    output_distribution="normal", 
                    random_state=self.random_state, 
                    **distorsion_params
                )
            ))
        pipeline.append(("scaling", StandardScaler()))
        pipeline.append((
            "pca", 
            PCA(n_components=pca_n_components, svd_solver='randomized', random_state=self.random_state)
        ))
        self.pipe = Pipeline(pipeline)


    def sample(self, X, sample_frac=1.0, random_state=42):
        """Случайная выборка"""
        n = X.shape[0]
        m = int(n * np.clip(sample_frac, a_min=0.0, a_max=1.0))
        if m < 1: m = 1
        rng = np.random.RandomState(random_state)
        rand_indices = rng.choice(n, m, replace=False)
        X_sample = X[rand_indices]
        return X_sample, rand_indices

    
    def fit(self, X, y=None):
        # Предварительные преобразования над подвыборкой
        if self.sample_frac < 1.0:
            self.X_sample, self.rand_indices = self.sample(X, sample_frac=self.sample_frac) 
        else:
            self.X_sample, self.rand_indices = X, np.arange(X.shape[0])
        self.X_pca = self.pipe.fit_transform(self.X_sample)

        # Кластеризация MiniBatchKMeans
        batch_size = min(2048, self.X_pca.shape[0])
        # rng = np.random.RandomState(self.random_state)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, 
            batch_size=batch_size, 
            random_state=self.random_state
        )
        self.kmeans.fit(self.X_pca)
        self.labels_ = self.kmeans.labels_
        self.is_fitted_ = True
        return self


    def predict(self, X):
        check_is_fitted(self, attributes=['is_fitted_'])
        X_transformed = self.pipe.transform(X)
        return self.kmeans.predict(X_transformed)


    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


    def hopkins_statistic(self, frac=0.2, n_jobs=None):
        """
        Статистика Хопкинса.
        - H ≈ 0.5: Данные случайны (нет структуры)
        - H → 1.0: Данные сильно кластеризованы (есть чёткие кластеры)
        - H → 0.0: Данные равномерно распределены (регулярная сетка)
        """
        check_is_fitted(self, attributes=['is_fitted_'])

        # Случайная выборка
        X_sample, _ = self.sample(self.X_pca, sample_frac=frac)

        # Генерация m равномерных случайных точек
        min_val = self.X_pca.min(axis=0)
        max_val = self.X_pca.max(axis=0)
        X_uniform = np.random.uniform(low=min_val, high=max_val, size=X_sample.shape)

        # wdist: расстояния до ближайших соседей в реальных данных для случайных точек из данных
        nnw = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=n_jobs)
        nnw.fit(self.X_pca)
        sample_dist, _ = nnw.kneighbors(X_sample)
        wdist = sample_dist[:, 1]  # Сосед №2, чтобы не взять ту же самую точку

        # udist: расстояния до ближайших соседей в реальных данных для равномерных точек
        nnu = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=n_jobs)
        nnu.fit(self.X_pca)
        uniform_dist, _ = nnu.kneighbors(X_uniform)
        udist = uniform_dist[:, 0] 

        # Статистика Хопкинса 
        sum_w = np.sum(wdist)
        sum_u = np.sum(udist)
        H = sum_u / (sum_w + sum_u)
        return H


    def stability(self, n_repeats=5, frac=0.8):
        """
        Оценка стабильности кластеризации через бутстрап/подвыборки и ARI.
        Идея: Если найденные кластеры отражают реальные паттерны, а не случайный шум, то они устойчивы к изменению выборки.

        0.85 - 1.00	Идеальная	В данных существуют четкие, плотные, устойчивые кластеры. Границы воспроизводятся почти на 100%.
        0.70 - 0.85	Хорошая	    Кластерная структура выражена хорошо, но на границах есть небольшое количество пограничных объектов.
        0.50 - 0.70	Умеренная	Кластеры сильно пересекаются или имеют сложную, несферическую форму.
        < 0.50	Нестабильная	Кластеров нет (однородное распределение/шум), алгоритм на каждом запуске нарезает пространство случайным образом.        
        """
        return cluster_stability(
            self.X_pca, 
            n_clusters=self.n_clusters, 
            sample_frac=frac, 
            random_state=self.random_state
        )
    

# ----------------------------------
# Альтернативный пайплайн по строкам
# ----------------------------------

def get_optimal_pca_components_l2(
        X, 
        n_components_eval=None, 
        n_iterations=20, 
        sample_frac=1.0, 
        random_state=42
):
    """
    Находит минимально необходимое число компонент PCA методом Хорна.
    Предварительно проводит L2-нормализацию по строкам (векторам).

    Параметры:
    - X: исходный датасет
    - n_components_eval: количество первых компонент для исследования
    - sample_frac: доля строк исходного датасета, которая используется для оценки
    """  
    # Если число компонент не указано, то уменьшить размерность минимум в 5 раз
    n_components_eval = n_components_eval if n_components_eval else int(X.shape[1]/5)
    
    pre_norm_l2 = Normalizer(norm="l2")  # Нормализует каждую СТРОКУ (картинку)
    X_trans = pre_norm_l2.fit_transform(X)
    horn = horn_parallel_analysis(
        X_trans, 
        n_components_eval=n_components_eval,
        n_iterations=n_iterations,
        sample_frac=sample_frac, 
        random_state=random_state
    )
    return horn['k_optimal']


def evaluate_pipeline_l2(X, pca_variance, hopkins_stat, n_clusters, random_state=42):
    """
    Функция оценки одной комбинации параметров для кластеризации
    """
    batch_size = min(2048, X.shape[0])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    # Расчет метрик в ПРЕОБРАЗОВАННОМ пространстве
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    # Доля наименьшего кластера
    min_cluster_pct = np.bincount(labels).min() / len(labels)
    
    # Стабильность кластеризации
    stab = cluster_stability(X, n_clusters=n_clusters)

    return {
        "pca_variance": pca_variance,
        "n_pca_components_actual": X.shape[1],
        "hopkins_stat": hopkins_stat,
        "n_clusters": n_clusters,
        "calinski_harabasz": ch,        # Чем ВЫШЕ, тем лучше
        "davies_bouldin": db,           # Чем НИЖЕ, тем лучше
        "stability": stab["stability_mean"],  # Чем ВЫШЕ, тем лучше [-0.5, 1.0]
        "stability_std": stab["stability_std"],
        "min_cluster_pct": min_cluster_pct
    }


def clustering_measure_l2(X, n_components: int, clusters: list, n_jobs=None, random_state=42):
    """
    Реализует пайплайн 
    1. Normalizer(norm="l2") по строкам,
    2. PCA с оптимальным количеством компонент, переданным в параметре n_components,
    3. Normalizer(norm="l2") по строкам,
    4. кластеризация методом MiniBatchKMeans.
    
    Перебирает варианты количества кластеров clusters, для каждого указывает
    - количество компонент и долю объясненной дисперсии в PCA,
    - индекс Калински-Харабаса, 
    - индекс Девиса-Болдина,
    - стабильность кластеризации,
    - долю наименьшего кластера.
    Возвращает результат в виде таблицы pandas Dataframe.

    Параметры:
    - X: исходный датасет.
    - n_components: Оптимальное количество компонент PCA.
    - n_clusters: Список или итератор с вариантами количества кластеров.
    """
    # Если n_jobs не указан, то использовать однозадачный режим (максимальная совместимость)
    n_jobs = n_jobs if n_jobs else 1

    # Пайплайн ТОЛЬКО для предобработки и сжатия
    pipe_pca = Pipeline([
        ("l2_norm_1", Normalizer(norm="l2")),
        ("pca", PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)),
        ("l2_norm_2", Normalizer(norm="l2"))
    ])
    
    # Трансформация и сжатие даных
    X_trans = pipe_pca.fit_transform(X)

    # Расчет статистики Хопкинса на преобразованных данных
    hstat = hopkins_statistic(X_trans, random_state=random_state)

    # Доля объясненной дисперсии
    pca_variance = float(np.sum(pipe_pca['pca'].explained_variance_ratio_))

    # Запуск параллельного перебора
    results = Parallel(n_jobs=min(n_jobs, len(clusters)), backend="loky")(
        delayed(evaluate_pipeline_l2)(X_trans, pca_variance, hstat, n_clusters, random_state) 
        for n_clusters in tqdm(clusters, desc="Перебор по сетке")
    )

    # Сбор результатов в DataFrame
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


class PipeKMeansClusterL2(ClusterMixin, BaseEstimator):
    def __init__(
        self, 
        pca_n_components, 
        n_clusters, 
        sample_frac=1.0, 
        random_state=42
    ):
        self.pca_n_components = pca_n_components
        self.n_clusters = n_clusters
        self.sample_frac = np.clip(sample_frac, a_min=0.0, a_max=1.0)
        self.random_state = random_state
        # rng = np.random.RandomState(random_state)

        self.X_sample = None
        self.rand_indices = None
        self.X_pca = None
        self.kmeans = None
        self.labels_ = None

        self.pipe = Pipeline([
            ("l2_norm_1", Normalizer(norm="l2")),  # Нормализует каждую СТРОКУ (картинку)
            ("pca", PCA(n_components=pca_n_components, svd_solver="randomized", random_state=42)),
            ("l2_norm_2", Normalizer(norm="l2"))   # Возвращает сжатые векторы на единичную сферу
        ])

    def sample(self, X, sample_frac=1.0, random_state=42):
        """Случайная выборка"""
        n = X.shape[0]
        m = int(n * np.clip(sample_frac, a_min=0.0, a_max=1.0))
        if m < 1: m = 1
        rng = np.random.RandomState(random_state)
        rand_indices = rng.choice(n, m, replace=False)
        X_sample = X[rand_indices]
        return X_sample, rand_indices

    
    def fit(self, X, y=None):
        # Предварительные преобразования над подвыборкой
        if self.sample_frac < 1.0:
            self.X_sample, self.rand_indices = self.sample(X, sample_frac=self.sample_frac) 
        else:
            self.X_sample, self.rand_indices = X, np.arange(X.shape[0])
        self.X_pca = self.pipe.fit_transform(self.X_sample)

        # Кластеризация MiniBatchKMeans
        batch_size = min(2048, self.X_pca.shape[0])
        # rng = np.random.RandomState(self.random_state)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, 
            batch_size=batch_size, 
            random_state=self.random_state
        )
        self.kmeans.fit(self.X_pca)
        self.labels_ = self.kmeans.labels_
        self.is_fitted_ = True
        return self


    def predict(self, X):
        check_is_fitted(self, attributes=['is_fitted_'])
        X_transformed = self.pipe.transform(X)
        return self.kmeans.predict(X_transformed)


    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


    def hopkins_statistic(self, frac=0.2, n_jobs=None):
        """
        Статистика Хопкинса.
        - H ≈ 0.5: Данные случайны (нет структуры)
        - H → 1.0: Данные сильно кластеризованы (есть чёткие кластеры)
        - H → 0.0: Данные равномерно распределены (регулярная сетка)
        """
        check_is_fitted(self, attributes=['is_fitted_'])

        # Случайная выборка
        X_sample, _ = self.sample(self.X_pca, sample_frac=frac)

        # Генерация m равномерных случайных точек
        min_val = self.X_pca.min(axis=0)
        max_val = self.X_pca.max(axis=0)
        X_uniform = np.random.uniform(low=min_val, high=max_val, size=X_sample.shape)

        # wdist: расстояния до ближайших соседей в реальных данных для случайных точек из данных
        nnw = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=n_jobs)
        nnw.fit(self.X_pca)
        sample_dist, _ = nnw.kneighbors(X_sample)
        wdist = sample_dist[:, 1]  # Сосед №2, чтобы не взять ту же самую точку

        # udist: расстояния до ближайших соседей в реальных данных для равномерных точек
        nnu = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=n_jobs)
        nnu.fit(self.X_pca)
        uniform_dist, _ = nnu.kneighbors(X_uniform)
        udist = uniform_dist[:, 0] 

        # Статистика Хопкинса 
        sum_w = np.sum(wdist)
        sum_u = np.sum(udist)
        H = sum_u / (sum_w + sum_u)
        return H


    def stability(self, n_repeats=5, frac=0.8):
        """
        Оценка стабильности кластеризации через бутстрап/подвыборки и ARI.
        Идея: Если найденные кластеры отражают реальные паттерны, а не случайный шум, то они устойчивы к изменению выборки.

        0.85 - 1.00	Идеальная	В данных существуют четкие, плотные, устойчивые кластеры. Границы воспроизводятся почти на 100%.
        0.70 - 0.85	Хорошая	    Кластерная структура выражена хорошо, но на границах есть небольшое количество пограничных объектов.
        0.50 - 0.70	Умеренная	Кластеры сильно пересекаются или имеют сложную, несферическую форму.
        < 0.50	Нестабильная	Кластеров нет (однородное распределение/шум), алгоритм на каждом запуске нарезает пространство случайным образом.        
        """
        return cluster_stability(
            self.X_pca, 
            n_clusters=self.n_clusters, 
            sample_frac=frac, 
            random_state=self.random_state
        )


# -------------------------------
# Визуализация
# -------------------------------

def get_kmeans_centroid_samples(data, cluster_centers, n_samples=3, metric='euclidean'):
    """
    Возвращает индексы ближайших объектов для каждого центроида
    
    Args:
        data (2D ndarray): Набор данных
        cluster_centers (2D ndarray): Центроиды
        n_samples (int): Количество точек
    """
    nn = NearestNeighbors(n_neighbors=n_samples, metric=metric)
    nn.fit(data)
    # Поиск ближайших объектов для каждого центроида
    # distances — матрицы расстояний, indices — матрицы индексов в исходном X
    distances, indices = nn.kneighbors(cluster_centers)
    return distances, indices


def get_hdbscan_best_cluster_samples(hdbscan_model, cluster_num, n_samples=3):
    """
    Возвращает индексы самых типичных объектов в кластере cluster_num
    
    Args:
        hdbscan_model: обученная модель hdbscan.HDBSCAN
        cluster_num: номер кластера
        n_samples: int, число самых сильных выбросов
    """
    labels = hdbscan_model.labels_
    probabilities = hdbscan_model.probabilities_

    # Маска выбранного кластера
    cluster_mask = (labels == cluster_num)

    if not np.any(cluster_mask):
        return np.array([], dtype=int)

    # Индексы всех объектов кластера в исходном массиве
    cluster_indices = np.where(cluster_mask)[0]

    # Вероятности только для объектов кластера
    cluster_subset_prob = probabilities[cluster_mask]

    # Самые высокие вероятности
    k = min(n_samples, len(cluster_subset_prob))
    top_k_local_idx = np.argsort(-cluster_subset_prob)[:k]
    # print(cluster_subset_prob[top_k_local_idx])

    return cluster_indices[top_k_local_idx]


def get_hdbscan_outliers_samples(hdbscan_model, n_samples):
    """
    Возвращает ndarray (1D) длины min(n, число_выбросов) с индексами n самых сильных 
    выбросов по оценке outlier_scores_.
    
    Args:
        hdbscan_model: обученная модель hdbscan.HDBSCAN
        n_samples: int, число самых сильных выбросов
    """
    labels = hdbscan_model.labels_
    outlier_scores = hdbscan_model.outlier_scores_
    
    # Маска выбросов (метка -1)
    outlier_mask = (labels == -1)
    
    if not np.any(outlier_mask):
        return np.array([], dtype=int)
    
    # Индексы всех выбросов в исходном массиве
    outlier_indices = np.where(outlier_mask)[0]
    
    # Scores только для выбросов
    outlier_subset_scores = outlier_scores[outlier_mask]
    
    # Самые сильные выбросы (n наибольших scores)
    k = min(n_samples, len(outlier_subset_scores))
    top_k_local_idx = np.argsort(-outlier_subset_scores)[:k]
    # print(outlier_subset_scores[top_k_local_idx])
    
    return outlier_indices[top_k_local_idx]


def match_labels(
    labels_a, 
    labels_b, 
    ignore_noise=True, 
    noise_label=-1, 
    unmatched_value=None
):
    """
    Сопоставление меток кластеров B к меткам кластеров A на основе Венгерского алгоритма.
    Максимизирует суммарное пересечение (overlap) между парами кластеров.

    Параметры:
    ----------
    labels_a : array-like, shape (n_samples,)
        Опорные метки кластеров (эталон).
    labels_b : array-like, shape (n_samples,)
        Метки кластеров, которые нужно переименовать.
    ignore_noise : bool, default=True
        Если True, шумовые точки исключаются из матрицы пересечений,
        а метка шума в B автоматически отображается в `noise_label`.
    noise_label : int, default=-1
        Идентификатор шума / выбросов.
    unmatched_value : int, str или None, default=None
        Значение для меток из B, которым не хватило пары в A (при K_B > K_A):
        - Если None: не сопоставленные кластеры получают новые уникальные ID.
        - Если int (например, -1): кластеры без пары отображаются в это значение.

    Возвращает:
    -----------
    mapping : dict
        Словарь вида {метка_B: сопоставленная_метка_A}.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError(
            f"Размеры массивов не совпадают: {labels_a.shape[0]} != {labels_b.shape[0]}"
        )

    # 1. Маска для исключения шума при расчете матрицы пересечений
    if ignore_noise:
        valid_mask = (labels_a != noise_label) & (labels_b != noise_label)
    else:
        valid_mask = np.ones(len(labels_a), dtype=bool)

    # Уникальные метки без шума
    unique_a = np.unique(labels_a[valid_mask])
    unique_b = np.unique(labels_b[valid_mask])

    mapping = {}

    # Шумовая метка по умолчанию отображается в шум
    if ignore_noise and (noise_label in labels_b):
        mapping[noise_label] = noise_label

    # Краевой случай: если нет валидных кластеров для сопоставления
    if len(unique_a) == 0 or len(unique_b) == 0:
        for b in np.unique(labels_b):
            if b not in mapping:
                mapping[b] = b
        return mapping

    # 2. Индексация произвольных меток в диапазоны 0..N_A-1 и 0..N_B-1
    a_to_idx = {val: idx for idx, val in enumerate(unique_a)}
    b_to_idx = {val: idx for idx, val in enumerate(unique_b)}

    idx_a = np.array([a_to_idx[val] for val in labels_a[valid_mask]])
    idx_b = np.array([b_to_idx[val] for val in labels_b[valid_mask]])

    # 3. Векторизованное построение прямоугольной матрицы пересечений (n_a x n_b)
    n_a, n_b = len(unique_a), len(unique_b)
    cost = np.zeros((n_a, n_b), dtype=np.int64)
    np.add.at(cost, (idx_a, idx_b), 1)  # Аналогично for a, b in zip(idx_a, idx_b): cost[a, b] += 1

    # 4. Венгерский алгоритм (линейная задача о назначениях)
    # Минимизовать -cost = максимизовать совпадение
    row_ind, col_ind = linear_sum_assignment(-cost)

    matched_b = set()
    for r, c in zip(row_ind, col_ind):
        orig_a = unique_a[r]
        orig_b = unique_b[c]
        mapping[orig_b] = orig_a
        matched_b.add(orig_b)

    # 5. Обработка меток B, оставшихся без пары (если K_B > K_A)
    all_unique_b = np.unique(labels_b)
    
    # Ищем следующий свободный положительный ID, чтобы избежать коллизий
    used_labels = set(mapping.values())
    if len(labels_a) > 0 and labels_a.max() >= 0:
        next_free_label = int(labels_a.max() + 1)
    else:
        next_free_label = 0

    while next_free_label in used_labels:
        next_free_label += 1

    for b in all_unique_b:
        if b not in mapping:
            if unmatched_value is not None:
                mapping[b] = unmatched_value
            else:
                mapping[b] = next_free_label
                used_labels.add(next_free_label)
                next_free_label += 1

    return mapping


def draw_images(img_path, img_list, nrows=1, ncols=3, figsize=(12, 5)):
    """Вывести изображения по индексу"""
    ncols = max(1, ncols)
    nrows = min(
        max(1, nrows),
        math.ceil(len(img_list)/ncols)
    )
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    p = 0
    for p in range(ncols*nrows):
        if p >= len(img_list): break
        img = plt.imread(os.path.join(img_path, img_list[p]))
        row = p // ncols
        col = p % ncols
        if nrows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        ax.imshow(img)
        ax.axis('off')
        p = p + 1
    plt.tight_layout()


def show_2d_spaces(df, hdbscan_labels, kmeans_labels, frac=1.0, palette='viridis'):
    """
    Рисует раскрашенные по кластерам 2D диаграммы для HDBSCAN и KMeaNS

    Параметры:
    hdbscan_labels: поле с метками HDBSCAN
    kmeans_labels: поле с метками KMeans
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Кластеры HDBSCAN
    sns.scatterplot(
        df[df[hdbscan_labels]>=0].sample(frac=frac), 
        x='x', y='y', 
        hue=hdbscan_labels,
        palette=palette,
        s=10, linewidths=0,
        ax=ax[0]
    )
    # Шум HDBSCAN (если есть)
    if sum(df[hdbscan_labels]<0) > 0:
        sns.scatterplot(
            df[df[hdbscan_labels]<0].sample(frac=frac), 
            x='x', y='y', 
            color='black',
            s=1, linewidths=0,
            ax=ax[0],
            label='-1'
        )
    ax[0].set(xticks=[], yticks=[], xlabel=None, ylabel=None, title='HDBSCAN')
    ax[0].legend(title='Кластер', loc='upper left')

    # Кластеры KMeans
    sns.scatterplot(
        df[df[kmeans_labels]>=0].sample(frac=frac), 
        x='x', y='y', 
        hue=kmeans_labels,
        palette=palette,
        s=10, linewidths=0,
        ax=ax[1]
    )
    ax[1].set(xticks=[], yticks=[], xlabel=None, ylabel=None, title='KMeans')
    ax[1].legend(title='Кластер', loc='upper left')
    plt.tight_layout()
