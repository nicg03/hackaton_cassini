from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import gym

class FloodManagementEnv(gym.Env):
    def __init__(self):
        super(FloodManagementEnv, self).__init__()
        
        # Azioni: ridurre il livello dell'acqua (es., pompaggio)
        self.action_space = spaces.Discrete(4)  # [0: nessuna azione, 1: pompa bassa, 2: pompa alta, 3: diga]
        
        # Stato: livello d'acqua, risorse rimanenti, danno accumulato
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),  # [livello minimo d'acqua, risorse, danno]
            high=np.array([100, 100, 1000]),  # [livello massimo d'acqua, risorse, danno massimo]
            dtype=np.float32
        )
        
        self.state = None
        self.reset()

        # Parametri aggiuntivi
        self.noise_std = 2  # Rumore del sensore (deviazione standard)
        self.resource_cost = [0, 10, 25, 40]  # Costo delle risorse per azione
        self.efficiency = [0, 0.8, 1.2, 5]  # Efficacia dell'azione (litri d'acqua rimossi)
        
    def reset(self):
        # Stato iniziale: livello acqua, risorse, danno
        self.state = np.array([50, 100, 0], dtype=np.float32)  # [livello iniziale, risorse, danno]
        return self.state
    
    def step(self, action):
        level, resources, damage = self.state
        
        # Introduzione del rumore (simula errori nei sensori)
        true_level = level
        level += np.random.normal(0, self.noise_std)
        
        # Vincolo sulle risorse: se finite, nessuna azione possibile
        if resources <= 0:
            action = 0
        
        # Applica l'azione
        if action in [1, 2, 3]:
            # Riduzione del livello d'acqua in base all'efficienza
            level -= self.efficiency[action] * 10  # 10 litri per unità d'efficienza
            # Costo delle risorse
            resources -= self.resource_cost[action]
        
        # Aggiorna il danno in base al livello d'acqua
        if level > 80:  # Soglia critica
            damage += 20 * (level - 80)  # Incremento non lineare
        
        # Precipitazioni casuali (stocasticità)
        precipitation = np.random.choice([0, 5, 10], p=[0.7, 0.2, 0.1])  # Probabilità: 70% nessuna pioggia
        level += precipitation
        
        # Imposta i limiti del sistema
        level = np.clip(level, 0, 100)
        resources = np.clip(resources, 0, 100)
        damage = np.clip(damage, 0, 1000)
        
        # Stato aggiornato
        self.state = np.array([level, resources, damage], dtype=np.float32)
        
        # Ricompensa: minimizzare il danno e conservare risorse
        reward = -damage - (100 - resources) * 0.1
        
        # Fine episodio: quando le risorse o i danni sono estremi
        done = resources <= 0 or damage >= 1000
        
        # Informazioni aggiuntive (debug)
        info = {
            "true_level": true_level,
            "precipitation": precipitation
        }
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        level, resources, damage = self.state
        print(f"Livello d'acqua: {level:.2f}, Risorse: {resources:.2f}, Danno: {damage:.2f}")

env = DummyVecEnv([lambda: FloodManagementEnv()])

# Creazione del modello PPO
model = PPO('MlpPolicy', env, learning_rate=0.0001, gamma=0.99, n_steps=2048, batch_size=64, ent_coef=0.01, verbose=1)

# Addestramento del modello
model.learn(total_timesteps=100000)

# Salvataggio del modello
model.save("flood_management_model")

# Caricamento del modello
model = PPO.load("flood_management_model")

# Ciclo di test
total_reward = 0

for episode in range(20):  # Testiamo per 10 episodi
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _states = model.predict(state)  # Predici l'azione
        state, reward, done, info = env.step(action)  # Esegui l'azione
        episode_reward += reward  # Aggiungi la ricompensa
        # env.render()  # Mostra lo stato dell'ambiente
        print(f"{action}, {state}")

    print(f"Episode {episode + 1} - Total Reward: {episode_reward}")

    total_reward += episode_reward

# Calcolare la ricompensa media
average_reward = total_reward / 20
print(f"Average Reward over 20 episodes: {average_reward}")
