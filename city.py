import networkx as nx
from heapq import heappush, heappop
import random
import matplotlib.pyplot as plt


class CityNetwork:
    def __init__(self, casual = True, num_nodes = 0, nodes = None, edges = None, connectivity=0.4) -> None:
        self.num_nodes = num_nodes
        self.nodes = nodes
        self.edges = edges
        self.connectivity = connectivity

        self.city = nx.Graph()
        if casual:
            self.generate_city()
        else:
            self.populate_city()
    
    def generate_city(self):
        for i in range(self.num_nodes):
            is_hospital = True if random.randint(0, 10) > 8 else False
            hospital = random.randint(10, 100) if is_hospital else 0
            self.city.add_node(i,
                        index = i,
                        is_hospital = False,
                        population = 100,
                        deaths = 0,
                        injuries = 0,
                        damage = 0,
                        priority = 0)
            
            if is_hospital:
                self.city.add_node(i + self.num_nodes,
                                    is_hospital = True,
                                    capacity = hospital)

                self.city.add_edge(i, i + self.num_nodes, weight = 0)

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < self.connectivity:
                    distance = random.randint(1, 10)
                    self.city.add_edge(i, j, weight=distance)

    def populate_city(self):
        for i, attr in enumerate(self.nodes):
            self.city.add_node(i, **attr)

        for u, v, w in self.edges:
            self.city.add_edge(u, v, weight=w)

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    @staticmethod
    def allocate_resources_with_priority(graph):
        for hospital in [n for n, d in graph.nodes(data=True) if d.get('is_hospital')]:
            capacity = graph.nodes[hospital]['capacity']
            available_capacity = capacity

            # Trova i nodi con feriti e le loro priorità
            injured_nodes = [
                (graph.nodes[node]['priority'],  # Priorità (ordinamento inverso)
                nx.shortest_path_length(graph, source=hospital, target=node, weight='weight'),  # Distanza
                graph.nodes[node]['injuries'],  # Feriti
                node)  # Nodo
                for node in graph.nodes
                if graph.nodes[node].get('injuries', 0) > 0 and node != hospital
            ]

            # Usa una coda con priorità (priorità, distanza, feriti) per allocare risorse
            heap = []
            for priority, path_length, injuries, node in injured_nodes:
                heappush(heap, (priority, path_length, -injuries, node))

            # Allocazione iterativa
            while heap and available_capacity > 0:
                priority, path_length, neg_injuries, node = heappop(heap)
                injuries = -neg_injuries  # Ripristina il valore positivo dei feriti

                path = nx.shortest_path(graph, source=hospital, target=node, weight='weight')

                # Calcola risorse assegnabili
                allocate = min(injuries, available_capacity)
                graph.nodes[node]['injuries'] -= allocate
                available_capacity -= allocate

                print(f"Ospedale {hospital} ha inviato {allocate} risorse a nodo {node} "
                    f"(Priorità: {priority}, Peso percorso: {path_length}, Feriti rimanenti: {graph.nodes[node]['injuries']}), Percorso: {path}")

            # Aggiorna la capacità rimanente dell'ospedale
            graph.nodes[hospital]['capacity'] = available_capacity
    
    @staticmethod
    def simulate_attack(city, center, radius):
        city_attacked = city
        for node in city_attacked.nodes:
            try:
                distance = nx.shortest_path_length(city_attacked, source=center, target=node, weight='weight')
                if city_attacked.nodes[node]['is_hospital']:
                    if distance <= radius:
                        damage = max(0, 100 - (distance / radius) * 100)
                        if damage >= 50:
                            city_attacked.nodes[node]['is_hospital'] = False
                            city_attacked.nodes[node]['capacity'] = 0
                    continue
                if distance <= radius:
                    damage = max(0, 100 - (distance / radius) * 100)
                    city_attacked.nodes[node]['damage'] = damage
                    if damage > 70:
                        city_attacked.nodes[node]['priority'] = 2
                    elif damage > 0:
                        city_attacked.nodes[node]['priority'] = 1
                    else:
                        city_attacked.nodes[node]['priority'] = 0

                    population = city_attacked.nodes[node]['population']
                    if damage > 70:
                        city_attacked.nodes[node]['deaths'] = int(population * 0.5)  # 50% morti
                        city_attacked.nodes[node]['injuries'] = int(population * 0.3)  # 30% feriti

                        city_attacked.nodes[node]['population'] -= int(population * 0.5) 
                    elif damage > 30:
                        city_attacked.nodes[node]['deaths'] = int(population * 0.1)  # 10% morti
                        city_attacked.nodes[node]['injuries'] = int(population * 0.6)  # 60% feriti

                        city_attacked.nodes[node]['population'] -= int(population * 0.1) 
                    else:
                        city_attacked.nodes[node]['deaths'] = int(population * 0.05)  # 5% morti
                        city_attacked.nodes[node]['injuries'] = int(population * 0.2)  # 20% feriti

                        city_attacked.nodes[node]['population'] -= int(population * 0.05)

            except nx.NetworkXNoPath:
                continue

        return city_attacked

    @staticmethod
    def draw_city(city):
        node_colors = [
            'green' if city.nodes[node].get('is_hospital') else 'blue'
            for node in city.nodes
        ]
        nx.draw(
            city,
            with_labels=True,
            node_color=node_colors,
            node_size=800,
            font_weight='bold'
        )
        plt.show()

    @staticmethod
    def print_city(city):
        for node in city.nodes:
            print(city.nodes[node])
        print('-----------------')