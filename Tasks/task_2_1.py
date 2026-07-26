import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



#   df.loc[rows,columns]  locates a section inside the matrix
#   .sum() adds up every value in the matrix
#   pd.concat(matrix,matrix) adds 2 matrix below one another
#   .unique() removes ano duplicants
###   a series is like a rows of dictionary items


#   grabs the file
current_dir = Path(__file__).resolve().parent / "archive/results.csv"
df = pd.read_csv(current_dir)



#   removing duplucants
key_cols = ['date', 'home_team', 'away_team']
df['team_pair'] = df.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
df = df.drop_duplicates(subset=['date', 'team_pair'], keep='first')







# counts goals (splits each team into a group with every goal they got, then sum the goals into 1 raw using .sum() and puts them into a series )
home_goals = df.groupby('home_team')['home_score'].sum()
away_goals = df.groupby('away_team')['away_score'].sum()
total_goals = home_goals.add(away_goals, fill_value=0)  #adds both series into 1 series 




#counts points
df['home_points'] = (df['home_score'] > df['away_score']) * 3 + (df['home_score'] == df['away_score']) * 1
df['away_points'] = (df['away_score'] > df['home_score']) * 3 + (df['home_score'] == df['away_score']) * 1



#counts points (same as goals, see above)
home_points = df.groupby('home_team')['home_points'].sum()
away_points = df.groupby('away_team')['away_points'].sum()
total_points = home_points.add(away_points, fill_value=0)



# it counts how many rows the team was in it, therefore calculating the games played
games_played = pd.concat([df['home_team'], df['away_team']]).value_counts() 



# returns a list for the teams who have won more than 100 games
qualified_teams = list(games_played[games_played > 200].index) 


#returns a series with each player goals per match "gpm" and points per match "ppm" and sort them in descending order
gpm = (total_goals[qualified_teams] / games_played[qualified_teams]).sort_values(ascending=False)      
ppm = (total_points[qualified_teams] / games_played[qualified_teams]).sort_values(ascending=False)



# more sorting
sorted_total_goals = total_goals.sort_values(ascending=False)
sorted_total_points = total_points.sort_values(ascending=False)



# gets the goals for every player
goalscorers_path = current_dir.parent / "goalscorers.csv"
gdf = pd.read_csv(goalscorers_path)
sorted_player_goals = gdf['scorer'].value_counts()

all_teams = list(pd.concat([df['home_team'], df['away_team']]).unique())




# grabs the shootouts file
shootouts_path = current_dir.parent / "shootouts.csv"
sdf = pd.read_csv(shootouts_path)


#setup the decade column & counting it
df["decade"] = df["date"].astype(str).str[:3] + "0"
decades_p = df["decade"].value_counts()


# gets the matches that were drawn  with only the important columns
draw_matches = df.loc[df["home_points"] == 1, ["home_team", "away_team", "date","tournament"]]

#   merges the results data (draw_matches) with the tournament matches (sdf) when they have the same columns ("home_team","away_team","date") and gets the tournament columns out to count it
tournament_count = pd.merge(draw_matches, sdf, on=["home_team","away_team","date"])["tournament"].value_counts()


# removes duplicants from the goalscorers data so the merge below doesnt blow up into extra rows (keeps only the first goal event per match)
first_gdf = gdf.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

# merges the goalscorers (first_gdf) with the results (df) on the matching columns and only keeps the columns we actually need
goalss = pd.merge(first_gdf,df, on=["date","home_team","away_team"])[[ "home_team", "away_team","home_score","away_score","team",]]


# grabs the rows where the home team scored (is in "team") but still lost the match (away_score bigger than home_score)
failure_1 = goalss.loc[(goalss["away_score"] >= goalss["home_score"]) & (goalss["home_team"] == goalss["team"])]
# grabs the rows where the away team scored (is in "team") but still lost the match (home_score bigger than away_score)
failure_2 = goalss.loc[(goalss["away_score"] <= goalss["home_score"]) & (goalss["away_team"] == goalss["team"])]

failure = pd.concat([failure_1,failure_2])
# counts the pethetic teams
failure_team = failure["team"].value_counts()













# total goals scored in a match (both teams combined)
df["match_goals"] = df["home_score"] + df["away_score"]



# how far apart the final score was (0 for a draw, bigger number = a blowout)
df["margin"] = (df["home_score"] - df["away_score"]).abs()

# was this match a draw ? (used below to get a % of matches that were draws per decade)
df["is_draw"] = (df["home_score"] == df["away_score"])




# groups everything by decade and averages it out
goals_per_match_by_decade = df.groupby("decade")["match_goals"].mean()
draw_rate_by_decade = df.groupby("decade")["is_draw"].mean() * 100  # *100 to make it read as a percentage
avg_margin_by_decade = df.groupby("decade")["margin"].mean()

era_comparison = df.groupby("decade").agg(
    goals_per_match=("match_goals", "mean"),
    draw_rate_pct=("is_draw", lambda s: s.mean() * 100),   # *100 to make it read as a percentage
    avg_winning_margin=("margin", "mean")
)







def print_top10(title, series, is_float=False):
    print(title)
    for i, (name, val) in enumerate(series.head(10).items(), start=1):
        formatted = f"{val:.2f}" if is_float else f"{int(val)}"
        print(f"{i}.{name:<20} : {formatted}")
    print("-----------------------")
 
 
# same idea as print_top10 but for a table with multiple columns, sorted by decade instead of ranked by value
def print_era(title, table):
    print(title)
    for decade, row in table.sort_index().iterrows():
        print(f"{decade} : goals/match={row['goals_per_match']:.2f}  draw_rate={row['draw_rate_pct']:.2f}%  avg_margin={row['avg_winning_margin']:.2f}")
    print("-----------------------")


# makes a bar chart for the top 10 of any series, drawn onto whichever subplot slot (ax) its given
def plot_top10(ax, title, series, ylabel="value"):
    top10 = series.head(10)
    ax.bar(top10.index.astype(str), top10.values, color="steelblue")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=7)


# makes 3 bar charts for the era_comparison table, one per stat, each drawn onto its own subplot slot (axes_slice = list of 3 axes)
def plot_era(axes_slice, table):
    table = table.sort_index()

    axes_slice[0].bar(table.index.astype(str), table["goals_per_match"], color="seagreen")
    axes_slice[0].set_title("Goals per match by decade", fontsize=9)
    axes_slice[0].tick_params(axis="x", rotation=45, labelsize=7)

    axes_slice[1].bar(table.index.astype(str), table["draw_rate_pct"], color="goldenrod")
    axes_slice[1].set_title("Draw rate % by decade", fontsize=9)
    axes_slice[1].tick_params(axis="x", rotation=45, labelsize=7)

    axes_slice[2].bar(table.index.astype(str), table["avg_winning_margin"], color="indianred")
    axes_slice[2].set_title("Avg winning margin by decade", fontsize=9)
    axes_slice[2].tick_params(axis="x", rotation=45, labelsize=7)


print_top10("SHOOTOUTS PER DECADE", decades_p)
print_top10("SHOOTOUTS CAUSED PER TOURNAMENT", tournament_count)
print_top10("TEAMS WHO SCORED BUT STILL LOST THE MATCH", failure_team)
print_era("ERA COMPARISON (GOALS PER MATCH / DRAW RATE / WINNING MARGIN BY DECADE)", era_comparison)

print_top10("MOST SCORES TEAM", sorted_total_goals)
print_top10("MOST SCORES PLAYER", sorted_player_goals)
print_top10("MOST POINTS TEAMS", sorted_total_points)
print_top10("MOST POINTS PER MATCH TEAMS", ppm, is_float=True)
print_top10("MOST GOALS PER MATCH TEAMS", gpm, is_float=True)
print_top10("MOST SHOOTOUTS PER DECADE", decades_p)


# builds 1 big window with a grid of subplots, then hands each chart its own slot instead of opening a new window per chart
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()  # turns the 4x3 grid into 1 flat list so we can grab slots by index (axes[0], axes[1], ...)

plot_top10(axes[0], "SHOOTOUTS PER DECADE", decades_p, ylabel="shootouts")
plot_top10(axes[1], "SHOOTOUTS CAUSED PER TOURNAMENT", tournament_count, ylabel="shootouts")
plot_top10(axes[2], "TEAMS WHO SCORED BUT STILL LOST", failure_team, ylabel="matches")
plot_era([axes[3], axes[4], axes[5]], era_comparison)   # takes up 3 slots, 1 per stat

plot_top10(axes[6], "MOST SCORES TEAM", sorted_total_goals, ylabel="goals")
plot_top10(axes[7], "MOST SCORES PLAYER", sorted_player_goals, ylabel="goals")
plot_top10(axes[8], "MOST POINTS TEAMS", sorted_total_points, ylabel="points")
plot_top10(axes[9], "MOST POINTS PER MATCH", ppm, ylabel="points/match")
plot_top10(axes[10], "MOST GOALS PER MATCH", gpm, ylabel="goals/match")

axes[11].axis("off")  # last grid slot is unused, so just hide it

plt.tight_layout()
plt.show()
