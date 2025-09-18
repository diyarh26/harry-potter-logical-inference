ids = ['214034621','213932338']


class GringottsController:

    def __init__(self, map_shape, harry_loc, initial_observations):
        self.rows, self.cols = map_shape

        self.harry_loc = harry_loc

        self.known_vaults = set()
        self.known_dragons = set()
        self.known_traps = set()
        self.suspected_traps = set()
        self.safe_tiles = set()
        self.collected_vaults = set()
        self.safe_tiles.add(harry_loc)

        self.visited = set()
        self.visited.add(harry_loc)
        self.path_history = [harry_loc]
        self.backward_path_history = set()
        self.sulfur_zones = set()

        self.process_observations(initial_observations)
        self.flag = 0
        self.vault_pos = None

    def process_observations(self, observations):
        """
        Processes initial or new observations and updates the knowledge base.
        """
        sulfur = False
        dragons = []

        for obs in observations:
            if obs[0] == "sulfur":
                sulfur = True
            elif obs[0] == "vault":
                if obs[1] not in self.collected_vaults:
                    self.known_vaults.add(obs[1])
            elif obs[0] == "dragon":
                dragons.append(obs[1])
                self.known_dragons.add(obs[1])

        if sulfur:
            self.mark_suspected_traps()
            self.suspected_traps.difference_update(self.safe_tiles)
            self.suspected_traps.difference_update(self.collected_vaults)
            self.sulfur_zones.add(self.harry_loc)
        else:
            self.mark_safe_neighbors()
            self.eliminate_trap_possibilities()
        if dragons:
            for dragon_loc in dragons:
                self.safe_tiles.discard(dragon_loc)
                self.suspected_traps.discard(dragon_loc)
                self.known_traps.add(dragon_loc)

    def eliminate_trap_possibilities(self):
        """
        Eliminates suspected trap locations based on negative evidence (not smelling sulfur).
        """
        if self.harry_loc not in self.sulfur_zones:  # Didn't smell sulfur at this location
            adjacent = self.get_adjacent_tiles(self.harry_loc)
            self.suspected_traps.difference_update(adjacent)

    def mark_safe_neighbors(self):
        """
        Marks the neighbors of Harry's current location as safe if no sulfur is detected.
        """
        row, col = self.harry_loc
        adjacent = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        for loc in adjacent:
            if self.is_within_bounds(loc):
                self.safe_tiles.add(loc)

    def mark_suspected_traps(self):
        """
        Updates the suspected traps based on the current location of Harry.
        Assumes traps are adjacent if sulfur is detected.
        """
        row, col = self.harry_loc
        adjacent = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        for loc in adjacent:
            if self.is_within_bounds(loc) and loc not in self.safe_tiles:
                self.suspected_traps.add(loc)

    def is_within_bounds(self, loc):
        """
        Checks if a location is within the bounds of the map.
        """
        row, col = loc
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_next_action(self, observations):
        """
        Determines the next action based on current observations and knowledge base.
        """

        self.process_observations(observations)  # Update knowledge based on observations

        # 1. Check if we're on a known vault and haven't collected it yet (flag == 0)
        if self.harry_loc in self.known_vaults and self.flag == 0:
            self.known_vaults.discard(self.harry_loc)
            self.collected_vaults.add(self.harry_loc)
            return ("collect",)

        # 2. Handle the flag states for destroying and collecting vault
        if self.flag == 1:
            next_move = self.vault_pos  # Move to vault location
            self.harry_loc = next_move
            self.visited.add(next_move)
            self.flag = 2  # Change flag state to collect
            self.path_history.append(next_move)
            return ("move", next_move)

        if self.flag == 2:
            self.flag = 0  # Change flag state to go back to the start
            self.collected_vaults.add(self.vault_pos)
            self.known_vaults.discard(self.vault_pos)
            self.vault_pos = None
            return ("collect",)

        # 3. Check if we detected both vault and sulfur
        vault_obs = next((obs for obs in observations if obs[0] == "vault"), None)
        sulfur_obs = next((obs for obs in observations if obs[0] == "sulfur"), None)

        if vault_obs and sulfur_obs:  # Both vault and sulfur detected
            vault_location = vault_obs[1]
            if vault_location not in self.known_traps and vault_location not in self.collected_vaults:
                self.suspected_traps.discard(vault_location)
                self.vault_pos = vault_location
                self.flag = 1  # Change flag state to move to the vault
                self.safe_tiles.add(vault_location)
                return ("destroy", vault_location)

        # 4. Try to move to the nearest safe, unvisited tile
        next_move = self.find_next_move()
        if next_move:
            if next_move in self.suspected_traps and next_move not in self.known_dragons:
                self.suspected_traps.remove(next_move)
                self.safe_tiles.add(next_move)
                return ("destroy", next_move)

            self.harry_loc = next_move
            self.visited.add(next_move)
            self.path_history.append(next_move)
            return ("move", next_move)

        # 5. Check for adjacent suspected traps (even if no sulfur observed)
        adjacent_suspects = [loc for loc in self.suspected_traps if
                             self.is_adjacent2(self.harry_loc, loc) and loc not in self.known_dragons]
        if adjacent_suspects:
            best_suspect = max(
                adjacent_suspects,
                key=lambda loc: (self.has_unvisited_neighbor(loc),
                                 -self.manhattan_distance(loc, self.get_nearest_unvisited_tile(loc)), loc[1],
                                 loc[0])

            )
            if best_suspect not in self.known_vaults:
                self.suspected_traps.discard(best_suspect)
                self.safe_tiles.add(best_suspect)
                return ("destroy", best_suspect)

        # 6. Backtrack to a previously visited safe tile
        backtrack_move = self.find_next_move_backtrack()
        if backtrack_move:
            self.harry_loc = backtrack_move
            self.path_history.append(backtrack_move)
            return ("move", backtrack_move)

        # 7. Default: Wait if no other action is possible
        return ("wait",)

    def get_nearest_unvisited_tile(self, loc):
        """
        Find the nearest unvisited tile to a given location, optimizing the search.
        Leverages get_adjacent_tiles to find adjacent tiles.
        """

        nearest_unvisited = None
        min_distance = float('inf')
        visited_around = set()
        queue = [(loc, 0)]

        while queue:
            current_loc, distance = queue.pop(0)
            if current_loc not in visited_around:
                if current_loc not in self.visited and self.is_within_bounds(current_loc):
                    if distance < min_distance:
                        min_distance = distance
                        nearest_unvisited = current_loc
                        return nearest_unvisited

                visited_around.add(current_loc)
                adjacent_tiles = self.get_adjacent_tiles(current_loc)
                for adjacent_tile in adjacent_tiles:
                    queue.append((adjacent_tile, distance + 1))
        return nearest_unvisited

    def find_next_move(self):
        """
        Finds the next move to the nearest safe, unvisited tile, prioritizing
        unexplored areas and known vaults.
        """
        # 1. Prioritize known, unvisited vaults
        if self.known_vaults:
            nearest_vault = min(
                self.known_vaults,
                key=lambda vault: (
                    self.manhattan_distance(self.harry_loc, vault),
                    vault[1],  # Tie-breaker: Row
                    vault[0],  # Tie-breaker: Column
                )
            )
            path = self.find_path(self.harry_loc, nearest_vault)
            if path and len(path) > 1:
                next_move = path[1]
                if next_move not in self.known_traps and next_move not in self.known_dragons:
                    return next_move

        # 2. BFS to find nearest safe, unvisited tile
        queue = [(self.harry_loc, 0)]
        visited_this_search = {self.harry_loc}

        while queue:
            (row, col), distance = queue.pop(0)

            adjacent = [
                (row - 1, col),
                (row + 1, col),
                (row, col + 1),
                (row, col - 1)
            ]

            for next_loc in adjacent:
                if (self.is_within_bounds(next_loc) and
                        next_loc not in visited_this_search):
                    if next_loc not in self.visited:
                        if self.is_adjacent2(self.harry_loc, next_loc):
                            if next_loc not in self.known_traps and next_loc not in self.known_dragons:
                                return next_loc
                    else:
                        if self.is_adjacent2(self.harry_loc, next_loc):
                            if next_loc not in self.known_traps and next_loc not in self.known_dragons:
                                queue.append((next_loc, distance + 1))
                                visited_this_search.add(next_loc)
        return None

    def manhattan_distance(self, loc1, loc2):
        row1, col1 = loc1
        row2, col2 = loc2
        return abs(row1 - row2) + abs(col1 - col2)

    def find_path(self, start, end):
        """
        Finds a path between two locations using breadth-first search.
        """
        if start == end:
            return [start]

        queue = [(start, [start])]
        visited = set()

        while queue:
            (row, col), path = queue.pop(0)
            current = (row, col)

            if current == end:
                return path

            visited.add(current)
            adjacent = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)
            ]
            for next_loc in adjacent:
                if self.is_adjacent2(current, next_loc) and (self.is_within_bounds(next_loc) and
                                                             next_loc not in self.known_traps and
                                                             next_loc not in self.known_dragons and
                                                             next_loc not in visited):
                    if next_loc == end:
                        return path + [next_loc]
                    queue.append((next_loc, path + [next_loc]))
        return None

    def find_next_move_backtrack(self):
        """
        Finds the next move by backtracking to a previously visited safe tile without looping.
        """
        self.backward_path_history.add(self.harry_loc)

        row, col = self.harry_loc
        adjacent = [
            (row - 1, col),
            (row, col - 1),
            (row + 1, col),
            (row, col + 1)
        ]
        # Sort adjacent tiles based on the number of unvisited neighbors (more first)
        adjacent.sort(key=lambda loc: (self.has_unvisited_neighbor(loc)), reverse=True)

        # First priority: Find a tile with unvisited neighbors
        for loc in adjacent:
            if (self.is_within_bounds(loc) and
                    loc in self.visited and
                    loc not in self.known_traps and
                    loc not in self.known_dragons and
                    loc not in self.backward_path_history and
                    self.is_adjacent2(self.harry_loc, loc)):
                return loc  # Move to this tile

        distant_unvisited_tile = self.get_distant_unvisited_tile()
        if distant_unvisited_tile:
            # Find a path to the distant tile
            path = self.find_path(self.harry_loc, distant_unvisited_tile)
            if path and len(path) > 1:
                return path[1]  # Move towards the distant tile

        self.backward_path_history.clear()
        return None

    def has_unvisited_neighbor(self, loc):
        """
        Checks if a tile has any unvisited, safe neighbors.
        """
        row, col = loc
        adjacent = [
            (row, col + 1),
            (row + 1, col),
            (row, col - 1),
            (row - 1, col)
        ]
        unvisited_neighbor_count = 0
        for neighbor in adjacent:
            if (self.is_within_bounds(neighbor) and
                    neighbor not in self.visited and
                    neighbor not in self.known_traps and
                    neighbor not in self.known_dragons):
                unvisited_neighbor_count += 1  # Found an unvisited, safe neighbor
        return unvisited_neighbor_count

    def is_adjacent2(self, loc1, loc2):
        """
        Checks if two locations are adjacent (not diagonally).
        """
        row1, col1 = loc1
        row2, col2 = loc2
        return abs(row1 - row2) + abs(col1 - col2) == 1

    def get_adjacent_tiles(self, loc):
        """
        Returns a list of adjacent tiles to the given location.
        """
        row, col = loc
        adjacent = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        return [tile for tile in adjacent if self.is_within_bounds(tile)]

    def get_distant_unvisited_tile(self):
        """
        Checks if there are unvisited tiles distant from the current location
        and returns the closest one, if any.
        """
        unvisited_tiles = [
            (row, col) for row in range(self.rows) for col in range(self.cols)
            if (row, col) not in self.visited and (row, col) not in self.known_traps and
               (row, col) not in self.known_dragons
        ]

        if not unvisited_tiles:
            return None

        # Sort by distance (descending) and choose the farthest
        sorted_unvisited_tiles = sorted(
            unvisited_tiles,
            key=lambda loc: self.manhattan_distance(self.harry_loc, loc),
            reverse=True
        )

        return sorted_unvisited_tiles[0]