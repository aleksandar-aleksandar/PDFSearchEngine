class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_node(self, node):
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []        

    def add_edge(self, start, end):
        if start in self.adjacency_list:
            self.adjacency_list[start].append(end)
        else:
            self.adjacency_list[start] = [end]

    def get_neighbors(self, node):
        return self.adjacency_list.get(node, [])

    def search(self, query):
        result = []
        for node in self.adjacency_list:
            with open(f'pages/page{node}.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                if query in content:
                    result.append(node)
        return result
    
    def __str__(self):
        output = "Graph Adjacency List:\n"
        for node, neighbors in self.adjacency_list.items():
            if neighbors:
                output += f"{node} -> {neighbors}\n"
        return output
