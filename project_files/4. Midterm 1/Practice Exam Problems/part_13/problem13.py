#!/usr/bin/env python
# coding: utf-8

# # Problem 13: Soccer Guru
# 
# _Version 1.5_

# Soccer season is on and teams need to start preparing for the World Cup 2022. We need your help as a **Soccer Guru** to analyse different statistics and come up with insights to help the teams prepare better.
# 
# This problem tests your understanding of Pandas and SQL concepts.
# 
# **Important note.** Due to a limitation in Vocareum's software stack, this notebook is set to use the Python 3.5 kernel (rather than a more up-to-date 3.6 or 3.7 kernel). If you are developing on your local machine and are using a different version of Python, you may need to adapt your solution before submitting to the autograder.
# 

# **Exercise 0** (0 points). Run the code cell below to load the data, which is a SQLite3 database containing results and fixtures of various soccer matches that have been played around the globe since 1980.
# 
# Observe that the code loads all rows from the table, `soccer_results`, contained in the database file, `prob0.db`.
# 
# > You do not need to do anything for this problem other than run the next two code cells and familiarize yourself with the resulting dataframe, which is stored in the variable `df`.

# In[1]:


import sqlite3 as db
import pandas as pd
from datetime import datetime
from collections import defaultdict
disk_engine = db.connect('file:prob0.db?mode=ro', uri=True)

def load_data():
    df = pd.read_sql_query("SELECT * FROM soccer_results", disk_engine) 
    return df


# In[2]:


#pd.options.mode.chained_assignment = None


# In[3]:


# Test: Exercise 0 (exposed)
df = load_data()
assert df.shape[0] == 22851, "Row counts do not match. Try loading the data again"
assert df.shape[1] == 9, "You don't have all the columns. Try loading the data again"
print("\n(Passed!)")
df.head()


# Each row of this dataframe is a game, which is played between a "home team" (column `home_team`) and an "away team" (`away_team`). The number of goals scored by each team appears in the `home_score` and `away_score` columns, respectively.

# **Exercise 1** (1 point): Write an **SQL query** find the ten (10) teams that have the highest average away-scores since the year 2000. Your query should satisfy the following criteria:
# 
# - It should return two columns:
#     * `team`: The name of the team
#     * `ave_goals`: The team's average number of goals **in "away" games.** An "away game" is one in which the team's name appars in `away_team` **and** the game takes place at a "non-neutral site" (`neutral` value equals `FALSE`).
# - It should only include teams that have played **at least 30 away matches**.
# - It should round the average goals value (`ave_goals`) to three decimal places.
# - It should only return the top 10 teams in descending order by average away-goals.
# - It should only consider games played since 2000 (including the year 2000).
# 
# Store your query string as the variable, `query_top10_away`, below. The test cell will run this query string against the input dataframe, `df`, defined above and return the result in a dataframe named `offensive_teams`. (See the test cell.)
# 
# > **Note.** The following exercises have hidden test cases and you'll be awarded full points for passing both the exposed and hidden test cases.

# In[4]:


query_top10_away = '''
    select
        away_team as team,
        round(avg(away_score), 3) as ave_goals
    from soccer_results
    where strftime('%Y', date) >= '2000' and neutral = 'FALSE'
    group by away_team
    having count(*) >= 30
    order by 2 desc
    limit 10
'''

print(query_top10_away)


# In[5]:


# Test: Exercise 1 (exposed)
offensive_teams = pd.read_sql_query(query_top10_away, disk_engine)
df_cols = offensive_teams.columns.tolist()
df_cols.sort()
desired_cols = ['team', 'ave_goals']
desired_cols.sort()
print(offensive_teams.head(10))
assert offensive_teams.shape[0] == 10, "Expected 10 rows but returned dataframe has {}".format(offensive_teams.shape[0])
assert offensive_teams.shape[1] == 2, "Expected 2 columns but returned dataframe has {}".format(offensive_teams.shape[1])
assert df_cols == desired_cols, "Column names should be: {}. Returned dataframe has: {}".format(desired_cols, df_cols)

tolerance = .001
team_4 = offensive_teams.iloc[3].team
team_4_ave = offensive_teams.iloc[3].ave_goals
desired_team_4_ave = 1.763
assert (team_4 == "England" and abs(team_4_ave - 1.763) <= .001), "Fourth entry is {} with average of {}. Got {} with average of {}".format("England", 1.76, team_4, team_4_ave)

print("\n(Passed!)")


# In[6]:


# Hidden test cell: exercise1_hidden

print("""
In addition to the tests above, this cell will include some hidden tests.
You will only know the result when you submit your solution to the
autograder.
""")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# **Exercise 2** (2 points): Suppose we are now interested in the top 10 teams having the best goal **differential**, between the years 2012 and 2018 (both inclusive). A team's goal differential is the difference between the total number of goals it scored and the total number it conceded across all games (in the requested years).
# 
# Complete the function, `best_goal_differential()`, below, so that it returns a pandas dataframe containing the top 10 teams by goal differential, sorted in descending order of differential. The dataframe should have two columns: `team`, which holds the team's name, and `differential`, which holds its overall goal differential.
# 
# > As a sanity check, you should find the Brazil is the number one team, with a differential of 152 during the selected time period of 2012-2018 (inclusive). It should be the first row of the returned dataframe.

# In[7]:


def best_goal_differential():
    
    def using_pandas():
        
        # filtering by year with sqlite instead of pandas since sqlite is faster
        #sql_query = '''
        #    select *
        #    from soccer_results
        #    where strftime('%Y', date) between '2012'and '2018'
        #'''
        #df_sub = pd.read_sql_query(sql_query, disk_engine)
        
        # filtering by year
        year = pd.to_datetime(df.date).dt.year
        df_sub = df[year.isin(range(2012, 2019))]
        
        # defining lambda to compute sum of diffs function
        # separately since it is going to be used more than once 
        diff_sum = lambda df, x, y: sum(df[x] - df[y])
        
        # defining variables since they will be used more than once
        a, b = 'home_score', 'away_score'
        
        # computing home_differential series
        home_differential = (
            df_sub
                .groupby('home_team')[a, b]
                .apply(diff_sum, a, b)
        )
        
        # computing away_differential series
        away_differential = (
            df_sub
                .groupby('away_team')[a, b]
                .apply(diff_sum, b, a)
        )
        
        # computing differential dataframe
        differential = (
            (home_differential + away_differential) # series summed
                .sort_values(ascending = False)[:10] # series sorted and top 10 taken
                .astype(int) # data type set to int
                .to_frame('differential') # series converted to dataframe with `differential` column
                .rename_axis('team') # index column named
                .reset_index() # new numeric index created; previous index turned into `team` column
        )
        
        return differential
    
    def using_sqlite():
        sql_query = '''
            with 
                sub as (
                    select *
                    from soccer_results
                    where strftime('%Y', date) between '2012'and '2018'
                ),
                home as (
                    select
                        home_team,
                        sum(home_score - away_score) as diff
                    from sub
                    group by 1
                ),
                away as (
                    select
                        away_team,
                        sum(away_score - home_score) as diff
                    from sub
                    group by 1
                )
            select
                away_team as team,
                sum(a.diff + h.diff) as differential
            from away a, home h
            where a.away_team = h.home_team 
            group by 1 
            order by 2 desc
            limit 10
        '''
        differential = pd.read_sql_query(sql_query, disk_engine)
        
        return differential
    
    return using_sqlite()


# In[8]:


# Test: Exercise 2 (exposed)

diff_df = best_goal_differential()
df_cols = diff_df.columns.tolist()
df_cols.sort()
desired_cols = ['team', 'differential']
desired_cols.sort()

assert isinstance(diff_df, pd.DataFrame), "Dataframe object not returned"
assert diff_df.shape[0] == 10, "Expected 10 rows but returned dataframe has {}".format(diff_df.shape[0])
assert diff_df.shape[1] == 2, "Expected 2 columns but returned dataframe has {}".format(diff_df.shape[1])
assert df_cols == desired_cols, "Column names should be: {}. Returned dataframe has: {}".format(desired_cols, df_cols)

best_team = diff_df.iloc[0].team
best_diff = diff_df.iloc[0].differential
assert (best_team == "Brazil" and best_diff == 152), "{} has best differential of {}. Got team {} having best differential of {}".format("Brazil", 152, best_team, best_diff)

print("\n(Passed!)")


# In[9]:


# Hidden test cell: exercise2_hidden

print("""
In addition to the tests above, this cell will include some hidden tests.
You will only know the result when you submit your solution to the
autograder.
""")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# **Exercise 3** (1 point). Complete the function, `determine_winners(game_df)`, below. It should determine the winner of each soccer game.
# 
# In particular, the function should take in a dataframe like `df` from above. It should return a new dataframe consisting of all the columns from that dataframe plus a new columnn called **`winner`**, holding the name of the winning team. If there is no winner for a particular game (i.e., the score is tied), then the `winner` column should containing the string, `'Draw'`. Lastly, the rows of the output should be in the same order as the input dataframe.
# 
# You can use any dataframe manipulation techniques you want for this question _(i.e., pandas methods or SQL queries, as you prefer)._
# 
# > You'll need the output dataframe from this exercise for the subsequent exercies, so don't skip this one!

# In[10]:


def determine_winners(game_df):
    
    #def winner_func(row):
    #    if row['home_score'] > row['away_score']:
    #        return row['home_team']
    #    elif row['home_score'] < row['away_score']:
    #        return row['away_team']
    #    return 'Draw'
    #
    #game_df['winner'] = game_df.apply(
    #    lambda row: winner_func(row), 
    #    axis = 1
    #)
    
    game_df['winner'] = game_df.apply(
        lambda row:
        row['home_team'] if row['home_score'] > row['away_score'] else (
            row['away_team'] if row['home_score'] < row['away_score'] else 'Draw'
        ),
        axis = 1
    )

    return game_df


# In[11]:


# Test: Exercise 3 (exposed)

game_df = pd.read_sql_query("SELECT * FROM soccer_results", disk_engine)
winners_df = determine_winners(game_df)

game_winner = winners_df.iloc[1].winner
assert game_winner == "Ghana", "Expected Ghana to be winner. Got {}".format(game_winner)

game_winner = winners_df.iloc[2].winner
assert game_winner == "Draw", "Match was Draw. Got {}".format(game_winner)

game_winner = winners_df.iloc[3].winner
assert game_winner == "Mali", "Expected Mali to be winner. Got {}".format(game_winner)

print("\n(Passed!)")


# In[12]:


# Hidden test cell: exercise3_hidden

print("""
In addition to the tests above, this cell will include some hidden tests.
You will only know the result when you submit your solution to the
autograder.
""")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# **Exercise 4** (3 points): Given a team, its _home advantage ratio_ is the number of home games it has won divided by the number of home games it has played. For this exercise, we'll try to answer the question, how important is the home advantage in soccer? It's importance is factored into draws for competitions, for example, teams wanting to play at home the second leg of the matches of great importance such as tournament knockouts. (_This exercise has a pre-requisite of finishing Exercise 3 as we'll be using the results of the dataframe from that exercise in this one._)
# 
# Complete the function, `calc_home_advantage(winners_df)`, below, so that it returns the top 5 countries, among those that have played at least 50 **home** games, having the highest home advantage ratio. It should return a dataframe with two columns, **`team`** and **`ratio`**, holding the name of the team and its home advantage ratio, respectively. The ratio should be rounded to three decimal places. The rows should be sorted in descending order of ratio. If there are two teams with the same winning ratio, the teams should appear in alphabetical order by name.
# 
# > **Note 0.** As with our definition of away-games, a team plays a home game if it is the home team (`home_team`) **and** the field is non-neutral (i.e., `neutral` is `FALSE`).
# >
# > **Note 1.** You should find, for example, that Brazil is the number two team, with a home advantage ratio of 0.773.

# In[13]:


def calc_home_advantage(winners_df):
    
    def using_pandas():

        import numpy as np
        
        df = (
            winners_df
                .query('neutral == "FALSE"')
                .groupby('home_team')
                .filter(lambda x: len(x) >= 50)
                .assign(home_win = lambda row: np.where(row['home_score'] > row['away_score'], 1, 0))
                .groupby('home_team')
                .apply(lambda df, x: df['home_win'].sum() / len(df), 'home_win')
                .sort_values(ascending = False)[:5]
                .round(3)
                .to_frame('ratio')
                .rename_axis('team')
                .reset_index()
        ) 
        
        return df
     
    def using_sqlite():
        
        conn = db.connect('home_advantage.db')
        winners_df.to_sql('soccer_results', conn, if_exists = 'replace', index = False)
        
        sql_query = '''
            select
                home_team as team,
                round(sum(case when home_score > away_score then 1 else 0 end) * 1.0 / count(*), 3) as ratio
            from soccer_results
            where neutral = 'FALSE'
            group by 1
            having count(*) >= 50
            order by 2 desc
            limit 5
        '''
        
        return pd.read_sql_query(sql_query, conn)
        
        conn.close()
    
    methods = [using_pandas(), using_sqlite()]

    return methods[0]


# In[14]:


# Test: Exercise 4 (exposed)
from IPython.display import display

win_perc = calc_home_advantage(winners_df)

print("The solution, according to you:")
display(win_perc)

df_cols = win_perc.columns.tolist()
df_cols.sort()
desired_cols = ['team', 'ratio']
desired_cols.sort()

assert win_perc.shape[0] == 5, "Expected 5 rows, got {}".format(win_perc.shape[0])
assert win_perc.shape[1] == 2, "Expected 2 columns, got {}".format(win_perc.shape[1])
assert df_cols == desired_cols, "Expected {} columns but got {} columns".format(desired_cols, df_cols)

tolerance = .001
sec_team = win_perc.iloc[1].team
sec_perc = win_perc.iloc[1].ratio

assert (sec_team == "Brazil" and abs(sec_perc - .773) <= tolerance), "Second team should be {} with ratio of {}. Got {} with ratio of {}".format("Brazil", .773, sec_team, sec_perc)

print("\n(Passed!)")


# In[15]:


# Hidden test cell: exercise4_hidden

print("""
In addition to the tests above, this cell will include some hidden tests.
You will only know the result when you submit your solution to the
autograder.
""")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# **Exercise 5** (3 points) Now, we've seen how much the home advantage plays in, let us see how the results have looked 
# like in the previous tournaments, for the specific case of the FIFA World Cup matches.
# 
# In particular, complete the function, `points_table(winners_df, wc_year)`, below, so that it does the following:
# - It should take as input a dataframe, `winners_df`, having a "winner" column like that produced in Exercise 3, as well as a target year, `wc_year`.
# - It should consider only games in the given target year. Furthermore, it should only consider games where the `tournament` column has the value `"FIFA World Cup"`.
# - It should construct and return a "points table". This table should have two columns, **`team`**, containing the team name, and **`points`**, containing a points tally has defined below.
# - To compute the points, give the team 3 points for every win, 1 point for every draw, and 0 points (no points) for a loss.
# - In case of a tie in the points, sort the teams alphabetically
# 
# As an example output, for the 1998 FIFA World Cup, the points table is:
# 
# | team        | points |
# |-------------|--------|
# | France      | 19     |
# | Croatia     | 15     |
# | Brazil      | 13     |
# | Netherlands | 12     |
# | Italy       | 11     |

# In[81]:


def points_table(winners_df, wc_year):
    
    def using_pandas():

        import numpy as np
        
        winners_df_sub = (
            winners_df
                .assign(year = lambda row: pd.to_datetime(row['date']).dt.year)
                .query("(tournament == 'FIFA World Cup') & (year == {})".format(wc_year))
                .assign(
                    home_team_points = lambda row: 
                        np.where(
                            row['home_team'] == row['winner'], 
                            3, 
                            np.where(
                                row['winner'] == 'Draw',
                                1,
                                0
                            )
                        )
                )
                .assign(
                    away_team_points = lambda row: 
                        np.where(
                            row['away_team'] == row['winner'], 
                            3, 
                            np.where(
                                row['winner'] == 'Draw',
                                1,
                                0
                            )
                        )
                )
                .filter(['home_team', 'away_team', 'home_team_points', 'away_team_points'])
        )

        # At this point we have a data frame `winners_df_sub `that has been filtered
        # and prepared for us to aggregate it. We can do it in a couple of methods from here.
        # We can choose one or the other. You will have to select which method in `return`
        # line of `agg_subset`. The function `agg_subset` has 2 sub-functions.
    
        def agg_subset():
            
            # Method #1 (Complicated)
            
            # Here is a method that renames the columns specifically so that the `wide_to_long`
            # method can be used to pivot the columns and then use a `groupby` to just sum the points.
            # However, it is a bit complicated.
            
            def agg1():
                
                winners_df_sub.columns = ['|'.join([y, x]) for x, y in winners_df_sub.columns.str.split('_', n = 1)]
                
                df = (
                    pd.wide_to_long(
                        winners_df_sub.reset_index(),
                        ['team', 'team_points'],
                        'index',
                        'name',
                        sep = '|',
                        suffix = '\w+'
                    )
                        .reset_index(drop = True)
                        .groupby('team')['team_points']
                        .sum()
                        .to_frame('points')
                        .sort_values(by = ['points', 'team'], ascending = False)
                        .reset_index()
                )
                
                return df

            
            # Method #2 (Simpler)
            
            # This method is simpler. Here we just create 2 series using a groupby and then we
            # put the series together using `pd.concat` and then sum them using a `groupby`. We
            # then turn the series into a dataframe.
            
            def agg2():
                
                home_team_summary = (
                    winners_df_sub
                        .groupby('home_team')['home_team_points']
                        .sum()
                )
        
        
                away_team_summary = (
                    winners_df_sub
                        .groupby('away_team')['away_team_points']
                        .sum()
                )
        
                df = (
                    pd.concat([home_team_summary, away_team_summary])
                        .groupby(level = 0)
                        .sum()
                        .to_frame('points')
                        .rename_axis('team')
                        .sort_values(by = ['points', 'team'], ascending = False)
                        .reset_index()
                )
                
                return df
            
            return [agg1(), agg2()][0]
        
        return agg_subset()
     
    def using_sqlite():
        
        conn = db.connect('wc_winners.db')
        winners_df.to_sql('soccer_results', conn, if_exists = 'replace', index = False)
        
        sql_query = '''
            with 
                sub as(
                    select 
                        *,
                        case
                            when home_team = winner then 3
                            when winner = 'Draw' then 1
                            else 0
                        end as home_points,
                        case
                            when away_team = winner then 3
                            when winner = 'Draw' then 1
                            else 0
                        end as away_points
                    from soccer_results
                    where strftime('%Y', date) = ? and tournament = 'FIFA World Cup'
                )
            select
                team,
                sum(points) as points
            from (
                select 
                    home_team as team,
                    sum(home_points) as points
                from sub
                group by 1
                union
                select
                    away_team as team,
                    sum(away_points) as points
                from sub
                group by 1 
            )
            group by 1
            order by 2 desc, 1
        '''
        
        return pd.read_sql_query(sql_query, conn, params = [str(wc_year)])
        
        conn.close()

    return [using_pandas(), using_sqlite()][1]


# In[80]:


# Test: Exercise 5 (exposed)


tbl_1998 = points_table(winners_df, 1998)

assert tbl_1998.iloc[0].team == "France"
assert tbl_1998.iloc[0].points == 19
assert tbl_1998.iloc[1].team == "Croatia"
assert tbl_1998.iloc[1].points == 15
assert tbl_1998.iloc[2].team == "Brazil"
assert tbl_1998.iloc[2].points == 13
assert tbl_1998.iloc[3].team == "Netherlands"
assert tbl_1998.iloc[3].points == 12
assert tbl_1998.iloc[4].team == "Italy"
assert tbl_1998.iloc[4].points == 11

print("\n(Passed!)")


# In[ ]:


# Hidden test cell: exercise5_hidden

print("""
In addition to the tests above, this cell will include some hidden tests.
You will only know the result when you submit your solution to the
autograder.
""")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# **Fin!** You’ve reached the end of this part. Don’t forget to restart and run all cells again to make sure it’s all working when run in sequence; and make sure your work passes the submission process. Good luck!
