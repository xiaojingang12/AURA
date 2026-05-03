import argparse
import http.client
import json
import os
import urllib.parse
from datetime import datetime, timedelta

from tqdm import tqdm


DEFAULT_CATEGORIES = ["business", "entertainment", "health", "science", "technology"]


def append_json_record(filename, record):
    if not os.path.isfile(filename):
        data = [record]
    else:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.append(record)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def adjust_date(date_str, days_to_add=2):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return (date_obj + timedelta(days=days_to_add)).strftime("%Y-%m-%d")


def fetch_news_batch(conn, access_key, date, category, offset, save_path):
    params = urllib.parse.urlencode(
        {
            "access_key": access_key,
            "date": date,
            "categories": category,
            "sort": "published_desc",
            "languages": "en,-ar,-de,-es,-fr,-it,-nl,-no,-pt,-ru,-zh",
            "limit": 1,
            "offset": offset,
        }
    )
    conn.request("GET", f"/v1/news?{params}")
    response = conn.getresponse()
    decoded_data = response.read().decode("utf-8")

    try:
        payload = json.loads(decoded_data)
    except json.JSONDecodeError as exc:
        print(f"JSON error: {exc}")
        return False

    if "error" in payload:
        print(f"API error: {payload}")
        return False

    print(f"Date: {date} | Category: {category}")
    for item in tqdm(payload["data"]):
        append_json_record(save_path, item)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch English news records from the Mediastack API.")
    parser.add_argument("--access-key", default=os.getenv("MEDIASTACK_ACCESS_KEY", ""), help="Mediastack access key.")
    parser.add_argument("--output-path", default="news_api.json", help="Output JSON file.")
    parser.add_argument("--start-date", default="2023-09-26", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2025-07-26", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, help="News categories to fetch.")
    parser.add_argument("--offset-count", type=int, default=2, help="Number of offsets to fetch per category/date.")
    parser.add_argument("--date-step-days", type=int, default=2, help="Days to advance after each category batch.")
    parser.add_argument("--host", default="api.mediastack.com", help="Mediastack API host.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.access_key:
        raise SystemExit("Missing Mediastack access key. Use --access-key or set MEDIASTACK_ACCESS_KEY.")

    conn = http.client.HTTPConnection(args.host)
    date = args.start_date
    result = True
    remove_file(args.output_path)

    while result:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        cutoff_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        if date_obj > cutoff_date:
            break

        for category in args.categories:
            for offset in range(args.offset_count):
                result = fetch_news_batch(conn, args.access_key, date, category, offset, args.output_path)
                if not result:
                    break
            if not result:
                break

        date = adjust_date(date, args.date_step_days)


if __name__ == "__main__":
    main()
