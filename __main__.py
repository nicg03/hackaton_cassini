from city import CityNetwork


cityNetwork = CityNetwork(True, num_nodes = 20)
city = cityNetwork.city

cityNetwork.print_city(city)
cityNetwork.simulate_attack(city, 2, 5)
cityNetwork.print_city(city)
cityNetwork.allocate_resources_with_priority(city)
cityNetwork.print_city(city)

