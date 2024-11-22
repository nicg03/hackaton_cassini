import numpy as np
import gym
from gym import spaces

class FloodManagementWithPriorityEnv(gym.Env):
    def __init__(self):
        super(FloodManagementWithPriorityEnv, self).__init__()
        
        # Azioni: gestire l'acqua per il livello globale, ospedali o strade
        self.action_space = spaces.Discrete(4)  # [0: niente, 1: pompa globale, 2: ospedali, 3: strade]
        
        # Stato: [livello globale, risorse, danno totale, livello acqua ospedali, livello acqua strade]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),  # [livello minimo, risorse, danno, acqua ospedali, acqua strade]
            high=np.array([100, 100, 1000, 100, 100]),  # [massimo livello, risorse, danno, ecc.]
            dtype=np.float32
        )
        
        self.state = None
        self.reset()

        # Parametri aggiuntivi
        self.noise_std = 2  # Rumore nei sensori
        self.resource_cost = [0, 10, 15, 5]  # Costo delle azioni
        self.efficiency = [0, 1.2, 1.5, 0.8]  # Efficacia delle azioni
        self.priorities = [0, 1, 2]  # Priorità [globale, ospedali, strade]

    def reset(self):
        # Stato iniziale: livello globale, risorse, danno totale, acqua ospedali, acqua strade
        self.state = np.array([50, 100, 0, 40, 60], dtype=np.float32)
        return self.state

    def step(self, action):
        global_level, resources, total_damage, hospital_level, road_level = self.state
        
        # Introduzione del rumore nei sensori
        true_global_level = global_level
        global_level += np.random.normal(0, self.noise_std)
        
        # Vincolo sulle risorse
        if resources <= 0:
            action = 0  # Nessuna azione se non ci sono risorse

        # Applica l'azione
        if action == 1:  # Azione globale
            global_level -= self.efficiency[action] * 10
        elif action == 2:  # Azione ospedali
            hospital_level -= self.efficiency[action] * 10
        elif action == 3:  # Azione strade
            road_level -= self.efficiency[action] * 10

        # Costo delle risorse
        resources -= self.resource_cost[action]
        
        # Incrementa danni basati sulle priorità
        if global_level > 80:
            total_damage += 10 * (global_level - 80) * self.priorities[0]
        if hospital_level > 70:
            total_damage += 20 * (hospital_level - 70) * self.priorities[2]  # Ospedali prioritari
        if road_level > 75:
            total_damage += 10 * (road_level - 75) * self.priorities[1]  # Strade meno prioritarie

        # Simula precipitazioni casuali
        precipitation = np.random.choice([0, 5, 10], p=[0.7, 0.2, 0.1])
        global_level += precipitation
        hospital_level += precipitation * 0.8  # Gli ospedali possono essere più protetti
        road_level += precipitation

        # Imposta i limiti del sistema
        global_level = np.clip(global_level, 0, 100)
        hospital_level = np.clip(hospital_level, 0, 100)
        road_level = np.clip(road_level, 0, 100)
        resources = np.clip(resources, 0, 100)
        total_damage = np.clip(total_damage, 0, 1000)
        
        # Stato aggiornato
        self.state = np.array([global_level, resources, total_damage, hospital_level, road_level], dtype=np.float32)
        
        # Ricompensa: ridurre i danni e conservare risorse
        reward = -total_damage - (100 - resources) * 0.1
        
        # Fine episodio
        done = resources <= 0 or total_damage >= 1000
        
        # Informazioni aggiuntive
        info = {
            "true_global_level": true_global_level,
            "precipitation": precipitation
        }
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        global_level, resources, total_damage, hospital_level, road_level = self.state
        print(f"Globale: {global_level:.2f}, Risorse: {resources:.2f}, Danno: {total_damage:.2f}, "
              f"Ospedali: {hospital_level:.2f}, Strade: {road_level:.2f}")
