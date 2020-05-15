from rlcard.utils import *
import random

class OpenSpiel2RLCard(object):
    ''' The wrapper class for transforming OpenSpiel Game object to RLCard Env object.
    '''
    def __init__(self,  config, game_object):
        self._game = game_object
        self.allow_step_back = config['allow_step_back']
        self.allow_raw_data = config['allow_raw_data']
        self.record_action = config['record_action']
        if self.record_action:
            self.action_recorder = []

        # Get the number of players/actions in this game
        self.player_num = self._game.num_players()
        self.action_num = self._game.num_distinct_actions()

        self._parents = []

        # A counter for the timesteps
        self.timestep = 0
    
    def reset(self):
        self._state = self._game.new_initial_state()
        self._sample_external_events()
        player = self._state.current_player()
        legal_act = self._state.legal_actions(player)
        #state = self._state.observation_tensor(player)
        state = self._extract_state(self._state)
        return state, player


    def step(self, action):
        _parent_state = self._state
        self._parents.append(_parent_state)
        self._state = self._state.child(action)
        self._sample_external_events()
        player = self._state.current_player()
        self.timestep += 1
        if self.is_over() == False: 
            #state = self._state.observation_tensor(player)
            state = self._extract_state(self._state)
            return state, player
        else:
            return self._extract_state(_parent_state)

    def step_back(self):
        self._state = self._parents.pop() 
        self._sample_external_events()
        player = self._state.current_player()
        #state = self._state.observation_tensor(player)
        state = self._extract_state(self._state)
        return state, player

    def run(self, is_training=False):
        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)
            # Environment steps
            next_state, next_player_id = self.step(action)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)
        return trajectories, payoffs

    def get_state(self, player_id):
        extracted_state = {}
        extracted_state['obs'] = np.array(self._state.observation_tensor(player_id))
        extracted_state['legal_actions'] = self._state.legal_actions(player_id)
        #print(extracted_state)
        return extracted_state

    def _extract_state(self, state):
        extracted_state = {}
        player_id = state.current_player()
        extracted_state['obs'] = np.array(state.observation_tensor(player_id))
        extracted_state['legal_actions'] = state.legal_actions(player_id)
        return extracted_state


    def set_agents(self, agents):
        self.agents = agents
        # If at least one agent needs raw data, we set self.allow_raw_data = True
        for agent in self.agents:
            if agent.use_raw:
                self.allow_raw_data = True
                break

    def is_over(self):
        return self._state.is_terminal()

    def get_payoffs(self):
        return np.array(self._state.returns())

    def get_player_id(self):
        return self._state.current_player()

    def _sample_external_events(self):
        while self._state.is_chance_node():
            outcome = self._chance_event_sampler(self._state)
            self._state.apply_action(outcome)

    def _chance_event_sampler(self, state):
        act = random.sample(state.legal_actions(state.current_player()), 1)[0]
        return act
