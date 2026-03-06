from collections import deque

def shortest_path(path, n):    
    if n == 1:
        return 0
    
    all_paths = []
    
    def dfs(node, path, red_count, blue_count, visited):
        if node == n:
            path_length = len(path) - 1 
            if path_length > 0:
                all_paths.append({
                    'path': path[:],
                    'length': path_length,
                    'red': red_count,
                    'blue': blue_count
                })
            return
        
        if node not in path:
            return
        
        for neighbor, color in path[node].items():
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                
                new_red = red_count + (1 if color == 'R' else 0)
                new_blue = blue_count + (1 if color == 'B' else 0)
                
                dfs(neighbor, path, new_red, new_blue, visited)
                
                path.pop()
                visited.remove(neighbor)

    dfs(1, [1], 0, 0, {1})
    
    if not all_paths:
        return -1

    min_cost = float('inf')
    best_path_info = None
    
    for path_info in all_paths:
        red = path_info['red']
        blue = path_info['blue']
        length = path_info['length']
        
        balance_cost = abs(red - blue)
        total_cost = length + balance_cost
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_path_info = {
                'path': path_info['path'],
                'original_length': length,
                'red': red,
                'blue': blue,
                'balance_cost': balance_cost,
                'total': total_cost
            }
    
    return min_cost if min_cost != float('inf') else -1

n, m = map(int, input().split())
paths = {}

for i in range(1, n + 1):
    paths[i] = {}

for _ in range(m):
    i,j,c = input().split()
    i,j = int(i), int(j)

    paths[i][j] = c
    paths[j][i] = c

print(shortest_path(paths, n))