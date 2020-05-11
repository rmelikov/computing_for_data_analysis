def points_table(winners_df, wc_year):
    def using_pandas():
        import numpy as np

        winners_df_sub = (
            winners_df
                .assign(year=lambda row: pd.to_datetime(row['date']).dt.year)
                .query("(tournament == 'FIFA World Cup') & (year == {})".format(wc_year))
                .assign(
                    home_team_points=lambda row:
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
                    away_team_points=lambda row:
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
                        sep='|',
                        suffix='\w+'
                    )
                        .reset_index(drop=True)
                        .groupby('team')['team_points']
                        .sum()
                        .to_frame('points')
                        .sort_values(by=['points', 'team'], ascending=False)
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
        winners_df.to_sql('soccer_results', conn, if_exists='replace', index=False)

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

        return pd.read_sql_query(sql_query, conn, params=[str(wc_year)])

        conn.close()

    return [using_pandas(), using_sqlite()][1]