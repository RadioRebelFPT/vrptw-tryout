import csv, json, os, time

import func_collection as fc

def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def run_solver(instance_data, time_limit_sec, use_local_search, random_seed=None):
    start = time.perf_counter()
    solution = fc.solve_vrptw(instance_data, time_limit_sec=time_limit_sec, use_local_search=use_local_search,
                              random_seed=random_seed,)
    runtime = time.perf_counter() - start
    return solution, runtime

if __name__ == "__main__":
    instance_path = "data/Gehring_Homberger/homberger_200_customer_instances/C1_2_1.TXT"
    instance_data = fc.extract_content(instance_path)

    os.makedirs("results", exist_ok=True)

    initial_solution, initial_runtime = run_solver(instance_data, time_limit_sec=1, use_local_search=False)
    final_solution, final_runtime = run_solver(instance_data, time_limit_sec=30, use_local_search=True)

    initial_row = {
        "instance": instance_data["name"],
        "vehicle_used": initial_solution["vehicle_used"],
        "total_distance": initial_solution["total_distance"],
        "runtime_sec": round(initial_runtime, 4),
        "solution_found": initial_solution["solution_found"],
    }
    final_row = {
        "instance": instance_data["name"],
        "vehicle_used": final_solution["vehicle_used"],
        "total_distance": final_solution["total_distance"],
        "runtime_sec": round(final_runtime, 4),
        "solution_found": final_solution["solution_found"],
    }

    write_csv("results/day1_initial.csv", [initial_row])
    write_csv("results/day1_final.csv", [final_row])

    routes_payload = {
        "instance": instance_data["name"],
        "coords": instance_data["coords"],
        "initial_routes": initial_solution["routes"],
        "final_routes": final_solution["routes"],
    }
    with open("results/c1_2_1_routes.json", "w") as file:
        json.dump(routes_payload, file, indent=2)

    baseline_count = 30
    baseline_time_limit = 5
    baseline_rows = []
    baseline_routes = []

    for seed in range(1, baseline_count + 1):
        solution, runtime = run_solver(instance_data, time_limit_sec=baseline_time_limit, use_local_search=True,
                                       random_seed=seed,)
        baseline_rows.append({
            "instance": instance_data["name"],
            "seed": seed,
            "vehicle_used": solution["vehicle_used"],
            "total_distance": solution["total_distance"],
            "runtime_sec": round(runtime, 4),
            "solution_found": solution["solution_found"],
        })
        baseline_routes.append(solution["routes"])

    baseline_fields = ["instance", "seed", "vehicle_used", "total_distance", "runtime_sec", "solution_found"]
    write_csv("results/baselines.csv", baseline_rows, baseline_fields)

    edge_labels = fc.build_edge_labels(baseline_routes)
    X, y, pairs = fc.build_edge_dataset(instance_data, edge_labels)
    model, stats = fc.train_xgb_edge_model(X, y, random_seed=42)
    scored_edges = fc.score_edges(model, X, pairs)

    top_k = 200
    top_edges = [[i, j, float(score)] for (i, j), score in scored_edges[:top_k]]
    with open("results/c1_2_1_ml_edges.json", "w") as file:
        json.dump({"instance": instance_data["name"], "coords": instance_data["coords"], "top_edges": top_edges}, file, indent=2)

    summary = {
        "instance": instance_data["name"],
        "baseline_count": baseline_count,
        "baseline_time_limit_sec": baseline_time_limit,
        "positive_edges": len(edge_labels),
        "total_edges": len(pairs),
        "train_size": stats["train_size"],
        "test_size": stats["test_size"],
        "pos_weight": round(stats["pos_weight"], 4),
    }
    write_csv("results/ml_summary.csv", [summary])

    print("Initial solution:", initial_row)
    print("Final solution:", final_row)
    print("Day 2 summary:", summary)