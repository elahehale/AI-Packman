def findPathToClosestDot(self, gameState):
    """
    Returns a path (a list of actions) to the closest dot, starting from
    gameState.
    """
    # Here are some useful elements of the startState
    limit = 1
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)
    fringe = util.Stack()
    curr_path = []
    flag = True
    curr_state = (startPosition,[])
    fringe.push((startPosition, []))
    visited = []
    while flag:
        if fringe.isEmpty():
            limit = limit + 1
            if limit > 100:
                break
            fringe = util.Stack()
            flag = True
            fringe.push((startPosition, []))
            visited = []

        curr_state = fringe.pop()
        curr_path = curr_state[1]
        curr_state = curr_state[0]
        if problem.isGoalState(curr_state):
            break;
        if curr_state not in visited and len(curr_path) < limit:
            visited.append(curr_state)
            successors = problem.getSuccessors(curr_state)
            for s in successors:
                if s[0] not in visited:
                    new_path = list(curr_path)
                    new_path.append(s[1])
                    fringe.push((s[0], new_path))
    if limit <= 100:
        new_prob = PositionSearchProblem(gameState, start=startPosition, goal=curr_state, warn=False, visualize=False)
        return search.bfs(new_prob)
    else:
        return []