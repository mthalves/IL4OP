
DISCRETE_AGENT_CFG = {
    "astar":{},
    "despot":{
        "max_depth":20,
        "max_it":1000,
        "kwargs":{
            "discount_factor":0.95,
            "num_scenarios":100,
            "lambda_reg":0.005,
        },
    },
    "ibpomcp":{
        "max_depth":20,
        "max_it":1000,
        "kwargs":{
            "discount_factor":0.95,
            "particle_revigoration":True,
            "k":100,
        },
    },
    "pomcp":{
        "max_depth":20,
        "max_it":1000,
        "kwargs":{
            "discount_factor":0.95,
            "particle_revigoration":True,
            "k":100,
        },
    },
    "tbrhopomcp":{
        "max_depth":20,
        "max_it":1000,
        "kwargs":{
            "discount_factor":0.95,
            "particle_revigoration":True,
            "k":100,
            "smallbag_size":10,
            "time_budget":2.0,
        },
    },
}

CONTINUOUS_AGENT_CFG = {
    "pomcpdpw":{
        "max_depth":20,
        "max_it":1000,
        "kwargs":{},
    },
    "pomcpow":{
        "max_depth":20,
        "max_it":1000,
        "kwargs":{},
    },
}

"""
    Select your execution/planning method changing the bellow lines
"""
DISCRETE_METHOD   = "ibpomcp"
CONTINUOUS_METHOD = "pomcpow"

DISCRETE_AGENT = {
    "name":DISCRETE_METHOD,
    "args":DISCRETE_AGENT_CFG[DISCRETE_METHOD]
}

CONTINUOUS_AGENT = {
    "name":CONTINUOUS_METHOD,
    "args":CONTINUOUS_AGENT_CFG[CONTINUOUS_METHOD]
}