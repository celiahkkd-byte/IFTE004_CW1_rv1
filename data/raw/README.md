This folder is only needed for the older teacher-data workflow. The default submission
configuration uses the included Alpha Vantage 5-minute data in:

    data/external/alpha_vantage_intraday_5min/combined_teacher_format/

If you want to run the original teacher-data version, place the teacher-provided zip file here:

    data4rv-20260512T132943Z-3-001.zip

or unzip it so that the text files are available as:

    data/raw/data4rv/AAPL.txt
    data/raw/data4rv/AMZN.txt
    data/raw/data4rv/JPM.txt

The package ignores macOS metadata files such as ._AAPL.txt.
