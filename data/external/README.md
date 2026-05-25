External public data are downloaded by scripts/02_download_external_data.py.

If network access is disabled, manually place compatible files here:
- vix_fred.csv      FRED VIXCLS with columns DATE,VIXCLS
- us3m_fred.csv     FRED DTB3 with columns DATE,DTB3
- epu_daily.csv     policyuncertainty.com All_Daily_Policy_Data.csv
- ads.csv           Philadelphia Fed ADS current vintage, normalized to date,ads
- hsi.csv           Hang Seng close prices or returns; code tries to infer columns

The code never invents missing external observations. It forward-fills macro series to equity trading days where appropriate and logs missing coverage.
