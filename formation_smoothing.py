def smooth_single_team(formations, min_persist=10):
    smoothed = []
    prev_valid = None
    counter = 0

    for i, f in enumerate(formations):
        if f == "Unknown":
            smoothed.append(prev_valid or "Unknown")
            continue

        if f == prev_valid:
            counter += 1
        else:
            counter = 1

        if counter >= min_persist:
            prev_valid = f

        smoothed.append(prev_valid or f)

    return smoothed

def fill_unknowns(formations):
    # Forward fill
    for i in range(1, len(formations)):
        if formations[i] == "Unknown":
            formations[i] = formations[i-1]
    # Backward fill
    for i in range(len(formations)-2, -1, -1):
        if formations[i] == "Unknown":
            formations[i] = formations[i+1]
    return formations

def smooth_formations_per_team(team_formations_per_frame, min_persist=10):
    # Separate formation sequences per team
    team1_seq = [frame.get(1, "Unknown") for frame in team_formations_per_frame]
    team2_seq = [frame.get(2, "Unknown") for frame in team_formations_per_frame]

    team1_seq = smooth_single_team(team1_seq, min_persist)
    team2_seq = smooth_single_team(team2_seq, min_persist)

    team1_seq = fill_unknowns(team1_seq)
    team2_seq = fill_unknowns(team2_seq)

    # Reconstruct frame-wise dicts
    smoothed_formations = []
    for t1, t2 in zip(team1_seq, team2_seq):
        smoothed_formations.append({1: t1, 2: t2})

    return smoothed_formations



def stabilize_formations(team_formations_per_frame, min_persist=2880):
    def stabilize_sequence(sequence):
        stable = []
        current = sequence[0]
        counter = 1
        candidate = current

        for i in range(1, len(sequence)):
            if sequence[i] == candidate:
                counter += 1
            else:
                candidate = sequence[i]
                counter = 1

            if counter >= min_persist:
                current = candidate

            stable.append(current)

        return stable

    # Split into sequences per team
    team1_seq = [frame.get(1, "Unknown") for frame in team_formations_per_frame]
    team2_seq = [frame.get(2, "Unknown") for frame in team_formations_per_frame]

    # Stabilize each team's formation sequence
    team1_seq = stabilize_sequence(team1_seq)
    team2_seq = stabilize_sequence(team2_seq)

    # Repack the stabilized results into frame dicts
    stabilized = []
    for f1, f2 in zip(team1_seq, team2_seq):
        stabilized.append({1: f1, 2: f2})

    return stabilized
