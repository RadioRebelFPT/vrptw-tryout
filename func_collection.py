from pathlib import Path
import math
import random
import xgboost as xgb
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Return a list of all .txt files under Gehring_Homberger and Solomon folders.
def get_txt_files_name(data_dir="data"):
    base = Path(data_dir)
    folders = [base / "Gehring_Homberger", base / "Solomon"]
    txt_files = []

    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".txt":
                txt_files.append(path)
    return sorted(txt_files)


# Parse a Solomon/Homberger file into structured arrays.
def extract_content(file_path):
    lines = Path(file_path).read_text().splitlines()

    # Extract problem name (c1_10_1, etc.).
    name = ""
    for line in lines:
        if line.strip():
            name = line.strip()
            break

    # Extract vehicle count and capacity.
    vehicle_count = None
    capacity = None
    for i, line in enumerate(lines):
        if "NUMBER" in line and "CAPACITY" in line:
            for j in range(i + 1, len(lines)):
                parts = lines[j].split()
                if len(parts) >= 2 and parts[0].isdigit():
                    vehicle_count = int(parts[0])
                    capacity = int(parts[1])
                    break
            if vehicle_count is not None:
                break

    # Extract customer data: coordinates, demands, time windows, service times.
    coords = []
    demands = []
    time_windows = []
    service_times = []

    in_customer_section = False
    for line in lines:
        if line.strip().upper().startswith("CUSTOMER"):
            in_customer_section = True
            continue
        if not in_customer_section:
            continue
        parts = line.split()
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        _, x, y, demand, ready, due, service = parts[:7]
        coords.append((int(x), int(y)))
        demands.append(int(demand))
        time_windows.append((int(ready), int(due)))
        service_times.append(int(service))

    return {
        "name": name,
        "vehicle_count": vehicle_count,
        "capacity": capacity,
        "coords": coords,
        "demands": demands,
        "time_windows": time_windows,
        "service_times": service_times,
    }


# Compute straight-line distance between two coordinates.
def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Create a full distance matrix for all nodes.
def distance_matrix(coords):
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(coords[i], coords[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

# Create OR-Tools index manager (node indexing + vehicle count).
def build_manager(instance_data):
    num_locations = len(instance_data["coords"])
    num_vehicles = instance_data["vehicle_count"]
    if num_vehicles is None:
        raise ValueError("vehicle_count is missing from instance data")
    return pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)

# Create the OR-Tools routing model.
def build_routing(manager):
    return pywrapcp.RoutingModel(manager)

# Register distance callback and return its index.
def add_distance_callback(routing, manager, coords):
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(euclidean_distance(coords[from_node], coords[to_node])))
    
    return routing.RegisterTransitCallback(distance_callback)

# Add vehicle capacity constraints.
def add_capacity_dimension(routing, manager, demands, capacity):
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [capacity] * routing.vehicles(), 
                                            True, "Capacity")
    return demand_callback_index

# Add time window constraints including travel + service time.
def add_time_dimension(routing, manager, coords, service_times, time_windows):
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = int(round(euclidean_distance(coords[from_node], coords[to_node])))
        return travel + int(service_times[from_node])

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    horizon = max(due for _, due in time_windows)
    routing.AddDimension(time_callback_index, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node, (ready, due) in enumerate(time_windows):
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(int(ready), int(due))

    for vehicle_id in range(routing.vehicles()):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        time_dim.CumulVar(start_index).SetRange(0, horizon)
        time_dim.CumulVar(end_index).SetRange(0, horizon)

    return time_callback_index


# Configure OR-Tools search parameters and runtime limit.
def set_search_params(time_limit_sec, first_solution_strategy="PATH_CHEAPEST_ARC", metaheuristic=None, random_seed=None,):
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = int(time_limit_sec)
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, 
                                             first_solution_strategy)
    if metaheuristic:
        params.local_search_metaheuristic = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)
    if random_seed is not None:
        try:
            params.random_seed = int(random_seed)
        except AttributeError:
            pass
    return params


# Solve the routing model with the given parameters.
def solve_model(routing, params):
    return routing.SolveWithParameters(params)

# Extract routes and total distance from an OR-Tools solution.
def extract_solution(routing, manager, solution, coords):
    routes = []
    total_distance = 0.0
    vehicle_used = 0

    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        route = [manager.IndexToNode(index)]
        route_distance = 0.0

        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(index)
            to_node = manager.IndexToNode(next_index)
            route_distance += euclidean_distance(coords[from_node], coords[to_node])
            index = next_index
            route.append(manager.IndexToNode(index))

        if len(route) > 2:
            vehicle_used += 1
        routes.append(route)
        total_distance += route_distance

    return {
        "routes": routes,
        "total_distance": total_distance,
        "vehicle_used": vehicle_used,
        "solution_found": True,
    }

# End-to-end VRPTW solve: build model, add constraints, solve, return routes.
def solve_vrptw(instance_data, time_limit_sec=60, use_local_search=True, first_solution_strategy="PATH_CHEAPEST_ARC",
                metaheuristic="GUIDED_LOCAL_SEARCH", random_seed=None,):
    manager = build_manager(instance_data)
    routing = build_routing(manager)

    distance_callback_index = add_distance_callback(routing, manager, instance_data["coords"])
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    add_capacity_dimension(routing, manager, instance_data["demands"], instance_data["capacity"], )
    add_time_dimension(routing, manager, instance_data["coords"], instance_data["service_times"], 
                       instance_data["time_windows"], )
    search_params = set_search_params(time_limit_sec, first_solution_strategy=first_solution_strategy,
                                       metaheuristic=metaheuristic if use_local_search else None, random_seed=random_seed, )

    solution = solve_model(routing, search_params)
    if solution is None:
        return {
            "routes": [],
            "total_distance": None,
            "vehicle_used": 0,
            "solution_found": False,
        }
    return extract_solution(routing, manager, solution, instance_data["coords"])

# Length of overlap between two time windows.
def time_window_overlap(window_a, window_b):
    start = max(window_a[0], window_b[0])
    end = min(window_a[1], window_b[1])
    return max(0, end - start)

# Build full feature vector for directed edge (i -> j).
def pair_features(instance_data, i, j):
    coords = instance_data["coords"]
    demands = instance_data["demands"]
    time_windows = instance_data["time_windows"]
    service_times = instance_data["service_times"]

    dist = euclidean_distance(coords[i], coords[j])
    depot = 0
    depot_dist_i = euclidean_distance(coords[i], coords[depot])
    depot_dist_j = euclidean_distance(coords[j], coords[depot])

    # Angle from depot to each node, used to capture geometric alignment.
    angle_i = math.atan2(coords[i][1] - coords[depot][1], coords[i][0] - coords[depot][0])
    angle_j = math.atan2(coords[j][1] - coords[depot][1], coords[j][0] - coords[depot][0])
    angle_diff = abs(angle_i - angle_j)

    overlap = time_window_overlap(time_windows[i], time_windows[j])
    slack_i = time_windows[i][1] - time_windows[i][0]
    slack_j = time_windows[j][1] - time_windows[j][0]

    return [
        dist,
        demands[i],
        demands[j],
        time_windows[i][0],
        time_windows[i][1],
        time_windows[j][0],
        time_windows[j][1],
        overlap,
        service_times[i],
        service_times[j],
        depot_dist_i,
        depot_dist_j,
        angle_i,
        angle_j,
        angle_diff,
        slack_i,
        slack_j,
    ]

# Collect all directed edges (i -> j) that appear in baseline routes.
def build_edge_labels(routes_list):
    labels = set()
    for routes in routes_list:
        for route in routes:
            for idx in range(len(route) - 1):
                i = route[idx]
                j = route[idx + 1]
                if i == 0 or j == 0:
                    continue
                labels.add((i, j))
    return labels

# Create ML dataset of all directed pairs with binary labels.
def build_edge_dataset(instance_data, edge_labels):
    num_nodes = len(instance_data["coords"])
    X = []
    y = []
    pairs = []
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i == j:
                continue
            pairs.append((i, j))
            X.append(pair_features(instance_data, i, j))
            y.append(1 if (i, j) in edge_labels else 0)
    return X, y, pairs

# Train XGBoost classifier to score edge compatibility.
def train_xgb_edge_model(X, y, random_seed=42):
    combined = list(zip(X, y))
    random.Random(random_seed).shuffle(combined)
    split = int(0.8 * len(combined))
    train = combined[:split]
    test = combined[split:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    pos = sum(y_train)
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0  # Handle class imbalance.

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "seed": int(random_seed),
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    try:
        model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, verbose_eval=False)
    except xgb.core.XGBoostError:
        params["tree_method"] = "hist"
        params["predictor"] = "auto"
        model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, verbose_eval=False)
    return model, {"train_size": len(train), "test_size": len(test), "pos_weight": scale_pos_weight}

# Score all directed pairs and return them sorted by probability.
def score_edges(model, X, pairs):
    dmatrix = xgb.DMatrix(X)
    scores = model.predict(dmatrix)
    scored = list(zip(pairs, scores))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
