import itertools


class DynamicSplitRouter:
    def __init__(self, tasks, num_stages, max_groups, enabled=False):
        self.tasks = list(tasks)
        self.num_stages = num_stages
        self.max_groups = list(max_groups)
        self.enabled = enabled
        self.stage_groups = {s: [list(self.tasks)] for s in range(num_stages)}

    def task_to_group(self, stage_idx):
        mapping = {}
        for gid, group in enumerate(self.stage_groups[stage_idx]):
            for task in group:
                mapping[task] = gid
        return mapping

    def can_split_stage(self, stage_idx):
        return self.enabled and stage_idx >= 2 and len(self.stage_groups[stage_idx]) < self.max_groups[stage_idx]

    @staticmethod
    def _group_score(group, sim):
        if len(group) <= 1:
            return 0.0
        total = 0.0
        for t in group:
            total += sum(sim.get((t, tt), 0.0) for tt in group if tt != t) / (len(group) - 1)
        return total

    def best_bisect(self, group, sim):
        if len(group) < 2:
            return None
        no_split = self._group_score(group, sim)
        best = None
        n = len(group)
        for r in range(1, n):
            for combo in itertools.combinations(group, r):
                if group[0] not in combo:
                    continue
                a = list(combo)
                b = [x for x in group if x not in combo]
                if not b:
                    continue
                score = self._group_score(a, sim) + self._group_score(b, sim)
                delta = score - no_split
                if best is None or delta > best[0]:
                    best = (delta, a, b)
        return best

    def apply_split(self, stage_idx, parent_group, a, b):
        groups = self.stage_groups[stage_idx]
        parent_idx = groups.index(parent_group)
        groups[parent_idx:parent_idx + 1] = [a, b]
        self.stage_groups[stage_idx] = groups
        return parent_idx, parent_idx + 1
