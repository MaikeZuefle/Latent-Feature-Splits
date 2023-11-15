from copy import deepcopy

import numpy as np


def is_subset_sum(sets, target, n, classes):
    assert len(target) == classes

    if classes == 2:
        DP_table = [
            [[False for i in range(target[1] + 1)] for i in range(target[0] + 1)]
            for i in range(n + 1)
        ]

        # If sum is 0, then answer is true
        for i in range(n + 1):
            DP_table[i][0][0] = True

        # Fill the DP_table table in bottom up manner
        for i in range(1, n + 1):
            for j1 in range(1, target[0] + 1):
                for j2 in range(1, target[1] + 1):
                    if j1 >= sets[i - 1][0] and j2 >= sets[i - 1][1]:
                        DP_table[i][j1][j2] = (
                            DP_table[i - 1][j1][j2]
                            or DP_table[i - 1][j1 - sets[i - 1][0]][j2 - sets[i - 1][1]]
                        )

                    else:
                        DP_table[i][j1][j2] = DP_table[i - 1][j1][j2]

        return DP_table[n][target[0]][target[1]], DP_table

    elif classes == 3:

        DP_table = np.zeros(
            (n + 1, target[0] + 1, target[1] + 1, target[2] + 1)
        ).astype(bool)

        # If sum is 0, then answer is true
        for i in range(n + 1):
            DP_table[i][0][0][0] = True

        # Fill the DP_table table in bottom up manner
        for i in range(1, n + 1):
            for j1 in range(1, target[0] + 1):
                for j2 in range(1, target[1] + 1):
                    for j3 in range(1, target[2] + 1):
                        if (
                            j1 >= sets[i - 1][0]
                            and j2 >= sets[i - 1][1]
                            and j3 >= sets[i - 1][2]
                        ):
                            DP_table[i][j1][j2][j3] = (
                                DP_table[i - 1][j1][j2][j3]
                                or DP_table[i - 1][j1 - sets[i - 1][0]][
                                    j2 - sets[i - 1][1]
                                ][j3 - sets[i - 1][2]]
                            )

                            if DP_table[i][j1][j2][j3] and (j1 + j2 + j3) == sum(
                                target
                            ):
                                for n2 in range(i, n + 1):
                                    DP_table[n2][j1][j2][j3] = True
                                return (
                                    DP_table[n][target[0]][target[1]][target[2]],
                                    DP_table,
                                )

                        else:
                            DP_table[i][j1][j2][j3] = DP_table[i - 1][j1][j2][j3]

        return DP_table[n][target[0]][target[1]][target[2]], DP_table

    else:
        raise Exception(
            f"Subset sum only implemented for 2 or three classes, not {classes}!"
        )


def get_closest_solution(target, DP_table):
    DP_table = np.array(DP_table)
    final_DP_table = DP_table[-1]
    sums = np.zeros(final_DP_table.shape)

    if len(target) == 2:
        for i in range(sums.shape[0]):
            for j in range(sums.shape[1]):
                sums[i, j] = i + j
        sums_masked = final_DP_table * sums
        difference = sum(target) - sums_masked.max()
        argmax = sums_masked.argmax()
        new_target = [argmax // sums_masked.shape[1], argmax % sums_masked.shape[1]]

    elif len(target) == 3:
        for i in range(sums.shape[0]):
            for j1 in range(sums.shape[1]):
                for j2 in range(sums.shape[2]):
                    sums[i, j1, j2] = i + j1 + j2

        sums_masked = final_DP_table * sums
        difference = sum(target) - sums_masked.max()
        argmax = sums_masked.argmax()

        new_target = np.unravel_index(sums_masked.argmax(), sums_masked.shape)

    return new_target, difference


def get_subset_solutions(DP_table, target, sets, n):
    if len(target) == 2:

        class StackItem:
            def __init__(self, i, j1, j2, take, togo):
                self.i = i  # row index in the dp table
                self.j1 = j1  # column1 index in the dp table
                self.j2 = j2  # column2 index in the dp table
                self.take = take  # Indices of tuples to include in the DP_table
                self.togo = togo  # Value "to go" until reaching the `target` sum

            def __repr__(self) -> str:
                return f"i: {self.i}, j1: {self.j1}, j2: {self.j2}, take: {self.take}, togo: {self.togo}"

        stack = []

        target = np.array(target)
        sets = np.array(sets)

        stack.append(
            StackItem(n - 1, target[0], target[1], [n - 1], target - sets[n - 1])
        )

        while len(stack) > 0:
            item = stack.pop()
            i, j1, j2, take, togo = item.i, item.j1, item.j2, item.take, item.togo

            j = np.array([j1, j2])
            next_j = j - sets[i]
            next_j1, next_j2 = next_j[0], next_j[1]

            if i > 0 and DP_table[i][j1][j2]:
                new_take = deepcopy(take)
                new_take[-1] = i - 1
                new_togo = togo + sets[i] - sets[i - 1]
                stack.append(StackItem(i - 1, j1, j2, new_take, new_togo))

            elif np.array_equal(togo, np.array([0, 0])):
                return [sets[t] for t in take]

            elif (
                i > 0
                and 0 <= next_j1
                and 0 <= next_j1
                and next_j1 < (target[0] + 1)
                and next_j2 < (target[1] + 1)
            ):
                if DP_table[i][next_j1][next_j2]:
                    new_take = deepcopy(take)
                    new_take.append(i - 1)
                    new_togo = togo - sets[i - 1]
                    stack.append(StackItem(i - 1, next_j1, next_j2, new_take, new_togo))

        return None

    if len(target) == 3:

        class StackItem:
            def __init__(self, i, j1, j2, j3, take, togo):
                self.i = i  # row index in the dp table
                self.j1 = j1  # column1 index in the dp table
                self.j2 = j2  # column2 index in the dp table
                self.j3 = j3  # column3 index in the dp table
                self.take = take  # Indices of tuples to include in the DP_table
                self.togo = togo  # Value "to go" until reaching the `target` sum

            def __repr__(self) -> str:
                return f"i: {self.i}, j1: {self.j1}, j2: {self.j2}, j3: {self.j3}, take: {self.take}, togo: {self.togo}"

        stack = []

        target = np.array(target)
        sets = np.array(sets)

        stack.append(
            StackItem(
                n - 1, target[0], target[1], target[2], [n - 1], target - sets[n - 1]
            )
        )

        while len(stack) > 0:
            item = stack.pop()
            i, j1, j2, j3, take, togo = (
                item.i,
                item.j1,
                item.j2,
                item.j3,
                item.take,
                item.togo,
            )

            j = np.array([j1, j2, j3])
            next_j = j - sets[i]
            next_j1, next_j2, next_j3 = next_j[0], next_j[1], next_j[2]

            if i > 0 and DP_table[i][j1][j2][j3]:
                new_take = deepcopy(take)
                new_take[-1] = i - 1
                new_togo = togo + sets[i] - sets[i - 1]
                stack.append(StackItem(i - 1, j1, j2, j3, new_take, new_togo))

            elif np.array_equal(togo, np.array([0, 0, 0])):
                return [sets[t] for t in take]

            elif (
                i > 0
                and 0 <= next_j1
                and next_j1 < (target[0] + 1)
                and next_j2 < (target[1] + 1)
                and next_j3 < (target[2] + 1)
            ):
                if DP_table[i][next_j1][next_j2][next_j3]:
                    new_take = deepcopy(take)
                    new_take.append(i - 1)
                    new_togo = togo - sets[i - 1]
                    stack.append(
                        StackItem(i - 1, next_j1, next_j2, next_j3, new_take, new_togo)
                    )

        return None


def get_subsets(sets, target, n, classes=2):
    is_sum, DP_table = is_subset_sum(sets, target, n, classes)
    closest_solution, difference = get_closest_solution(target, DP_table)
    subset_solution = get_subset_solutions(DP_table, closest_solution, sets, n)
    return subset_solution, closest_solution, difference
