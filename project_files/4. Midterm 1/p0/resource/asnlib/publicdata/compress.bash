#!/usr/bin/env bash

zip -9Xr mt2p0-sp20-data.zip \
    compress.bash \
    locales.json us_states.csv \
    covid19/*.{csv,txt} update.bash \
    us-flights/{L_*.csv,README.md,us-flights-2019--86633396_T_ONTIME_REPORTING.csv}
