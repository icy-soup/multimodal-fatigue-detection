"""GA-PSO 联合优化

GA（遗传算法）用于特征选择，PSO（粒子群优化）用于超参调优。
两阶段顺序执行：GA 选出最优特征子集 → PSO 在该子集上调优 XGBoost 参数。

为避免过拟合，适应度使用 3-fold CV 评估。
最终用 LOSO 验证优化效果。
"""

import numpy as np
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

# ─── GA 参数 ──────────────────────────────────────────────
GA_POP_SIZE = 30          # 种群大小
GA_CROSSOVER_RATE = 0.9   # 交叉率
GA_MUTATION_RATE = 0.05   # 变异率
GA_N_GENERATIONS = 40     # 进化代数
N_FEATURES = 92           # 特征总数

# ─── PSO 参数 ─────────────────────────────────────────────
PSO_N_PARTICLES = 15
PSO_MAX_ITER = 20
PSO_W = 0.7               # 惯性权重
PSO_C1 = 1.4              # 认知系数
PSO_C2 = 1.4              # 社会系数

# XGBoost 超参搜索空间
HP_BOUNDS = {
    'n_estimators': (50, 300),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
}
HP_NAMES = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']


# ═══════════════════════════════════════════════════════════
#  第一阶段: GA 特征选择
# ═══════════════════════════════════════════════════════════

class GASelector:
    """遗传算法特征选择器"""

    def __init__(self, X, y, n_features=N_FEATURES,
                 pop_size=GA_POP_SIZE, crossover_rate=GA_CROSSOVER_RATE,
                 mutation_rate=GA_MUTATION_RATE, n_generations=GA_N_GENERATIONS,
                 random_state=42):
        self.X = X
        self.y = y
        self.n_features = n_features
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.rng = np.random.RandomState(random_state)

        self.pop = None
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = 0

    def _init_population(self):
        """初始化种群：每个个体是 n_features 位的二进制编码"""
        self.pop = self.rng.randint(2, size=(self.pop_size, self.n_features))
        # 确保每个个体至少选1个特征
        for i in range(self.pop_size):
            if self.pop[i].sum() == 0:
                self.pop[i, self.rng.randint(self.n_features)] = 1

    def _fitness(self, individual):
        """计算个体适应度

        用 XGBoost 在筛选后的特征上做 3-fold CV 的 F1 作为适应度。
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        from sklearn.metrics import f1_score

        mask = individual.astype(bool)
        if mask.sum() == 0:
            return 0.0

        X_sub = self.X[:, mask]

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_sub, self.y):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric='logloss', random_state=42, verbosity=0)
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_val_s)
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

        return np.mean(f1_scores)

    def _evaluate_population(self):
        """评估整个种群"""
        fitness = np.array([self._fitness(ind) for ind in self.pop])
        self.fitness_history.append(fitness.mean())
        return fitness

    def _select(self, fitness):
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size):
            i, j = self.rng.randint(self.pop_size, size=2)
            winner = self.pop[i] if fitness[i] > fitness[j] else self.pop[j]
            selected.append(winner.copy())
        return np.array(selected)

    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if self.rng.rand() < self.crossover_rate:
            point = self.rng.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual):
        """位翻转变异"""
        for i in range(self.n_features):
            if self.rng.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        # 确保至少选1个
        if individual.sum() == 0:
            individual[self.rng.randint(self.n_features)] = 1
        return individual

    def evolve(self, verbose=True):
        """运行特征选择"""
        self._init_population()

        for gen in range(self.n_generations):
            fitness = self._evaluate_population()
            gen_best_idx = np.argmax(fitness)
            gen_best_f = fitness[gen_best_idx]

            if gen_best_f > self.best_fitness:
                self.best_fitness = gen_best_f
                self.best_individual = self.pop[gen_best_idx].copy()

            if verbose:
                n_selected = int(self.best_individual.sum())
                print(f"  Gen {gen+1:2d}/{self.n_generations} | "
                      f"best F1={self.best_fitness:.4f} | "
                      f"selected={n_selected}/{self.n_features}")

            # 选择 → 交叉 → 变异
            selected = self._select(fitness)
            new_pop = []
            for i in range(0, self.pop_size, 2):
                c1, c2 = self._crossover(selected[i], selected[(i + 1) % self.pop_size])
                new_pop.append(self._mutate(c1))
                new_pop.append(self._mutate(c2))
            self.pop = np.array(new_pop[:self.pop_size])

        return self.best_individual, self.best_fitness


# ═══════════════════════════════════════════════════════════
#  第二阶段: PSO 超参优化
# ═══════════════════════════════════════════════════════════

class PSOTuner:
    """粒子群优化超参调优器"""

    def __init__(self, X, y, n_particles=PSO_N_PARTICLES,
                 max_iter=PSO_MAX_ITER, w=PSO_W, c1=PSO_C1, c2=PSO_C2,
                 random_state=42):
        self.X = X
        self.y = y
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.RandomState(random_state)

        self.n_dims = len(HP_NAMES)
        # 粒子位置 (归一化到 [0,1])
        self.positions = None
        self.velocities = None
        self.pbest = None
        self.pbest_val = None
        self.gbest = None
        self.gbest_val = 0
        self.history = []

    def _denormalize(self, pos_norm):
        """将归一化位置映射到实际超参值"""
        hp = {}
        for i, name in enumerate(HP_NAMES):
            lo, hi = HP_BOUNDS[name]
            val = lo + pos_norm[i] * (hi - lo)
            if name in ('n_estimators', 'max_depth'):
                val = int(round(val))
            hp[name] = val
        return hp

    def _fitness(self, pos_norm):
        """评估粒子位置的适应度 (3-fold CV F1)"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        from sklearn.metrics import f1_score

        hp = self._denormalize(pos_norm)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, X_val = self.X[train_idx], self.X[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            clf = XGBClassifier(
                n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
                learning_rate=hp['learning_rate'], subsample=hp['subsample'],
                colsample_bytree=hp['colsample_bytree'],
                eval_metric='logloss', random_state=42, verbosity=0)
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_val_s)
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

        return np.mean(f1_scores)

    def optimize(self, verbose=True):
        """运行 PSO 优化"""
        self.positions = self.rng.rand(self.n_particles, self.n_dims)
        self.velocities = self.rng.uniform(-0.1, 0.1, (self.n_particles, self.n_dims))
        self.pbest = self.positions.copy()
        self.pbest_val = np.array([self._fitness(p) for p in self.positions])

        best_idx = np.argmax(self.pbest_val)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_val = self.pbest_val[best_idx]

        for iteration in range(self.max_iter):
            r1 = self.rng.rand(self.n_particles, self.n_dims)
            r2 = self.rng.rand(self.n_particles, self.n_dims)

            # 速度更新
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.pbest - self.positions) +
                               self.c2 * r2 * (self.gbest - self.positions))
            # 速度限制
            max_v = 0.2
            self.velocities = np.clip(self.velocities, -max_v, max_v)

            # 位置更新
            self.positions = np.clip(self.positions + self.velocities, 0, 1)

            # 适应度更新
            fitness = np.array([self._fitness(p) for p in self.positions])

            improved = fitness > self.pbest_val
            for i in range(self.n_particles):
                if improved[i]:
                    self.pbest[i] = self.positions[i].copy()
                    self.pbest_val[i] = fitness[i]

            current_best = fitness.max()
            if current_best > self.gbest_val:
                self.gbest_val = current_best
                self.gbest = self.positions[np.argmax(fitness)].copy()

            self.history.append(self.gbest_val)

            if verbose:
                hp = self._denormalize(self.gbest)
                print(f"  Iter {iteration+1:2d}/{self.max_iter} | "
                      f"best F1={self.gbest_val:.4f} | "
                      f"params={hp['n_estimators']}e/{hp['max_depth']}d/"
                      f"{hp['learning_rate']:.2f}lr/{hp['subsample']:.2f}sub")

        return self._denormalize(self.gbest), self.gbest_val


# ═══════════════════════════════════════════════════════════
#  联合优化入口
# ═══════════════════════════════════════════════════════════

def run_ga_pso(X, y, verbose=True):
    """运行 GA-PSO 联合优化

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 92)
    y : np.ndarray, shape (n_samples,)

    Returns
    -------
    result : dict with selected_features, best_params, cv_f1
    """
    print("\n" + "=" * 60)
    print("Phase 1: GA 特征选择")
    print("=" * 60)

    ga = GASelector(X, y)
    best_mask, ga_f1 = ga.evolve(verbose=verbose)

    n_selected = int(best_mask.sum())
    print(f"\nGA 完成: 选中 {n_selected}/{X.shape[1]} 个特征, CV F1={ga_f1:.4f}")

    X_selected = X[:, best_mask.astype(bool)]

    print("\n" + "=" * 60)
    print("Phase 2: PSO 超参优化 (在选中特征上)")
    print("=" * 60)

    pso = PSOTuner(X_selected, y)
    best_params, pso_f1 = pso.optimize(verbose=verbose)

    print(f"\nPSO 完成: best F1={pso_f1:.4f}")
    print(f"最佳参数: {best_params}")

    return {
        'selected_mask': best_mask,
        'n_selected': n_selected,
        'selected_features': np.where(best_mask)[0].tolist(),
        'best_params': best_params,
        'ga_cv_f1': ga_f1,
        'pso_cv_f1': pso_f1,
    }
