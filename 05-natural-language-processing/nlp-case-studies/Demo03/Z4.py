import csv
import os
import random
import time

import requests


def build_headers():
    cookie = os.environ.get("WEIBO_COOKIE", "").strip()
    if not cookie:
        raise RuntimeError(
            "WEIBO_COOKIE is required for authorized collection; "
            "do not place session credentials in source code."
        )

    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/123.0 Safari/537.36"
        ),
        "Cookie": cookie,
        "Referer": "https://weibo.com",
    }


def get_mid_from_bid(bid, headers):
    url = f"https://weibo.com/ajax/statuses/show?id={bid}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("mid")


def get_comments(mid, headers, max_id=0):
    url = (
        "https://weibo.com/ajax/statuses/buildComments?"
        f"flow=0&is_reload=1&id={mid}&is_show_bulletin=2&count=20&max_id={max_id}"
    )
    resp = requests.get(url, headers=headers, timeout=30)

    if resp.status_code != 200:
        print("请求失败：", resp.status_code)
        return None

    return resp.json()


def crawl_weibo_comments(bid, save_path="weibo_comments.csv", max_pages=20):
    headers = build_headers()
    mid = get_mid_from_bid(bid, headers)
    print("微博 mid 获取成功：", mid)

    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["用户名", "评论内容", "点赞数", "评论时间"])

        max_id = 0

        for page in range(max_pages):
            print(f"正在爬取第 {page + 1} 页评论（max_id={max_id}）...")

            data = get_comments(mid, headers, max_id)
            if not data:
                print("无数据或被反爬，终止。")
                break

            comments = data.get("data", [])
            if not comments:
                print("无更多评论，结束。")
                break

            for comment in comments:
                user = comment["user"]["screen_name"]
                text = comment["text_raw"]
                like = comment["like_counts"]
                time_str = comment["created_at"]
                writer.writerow([user, text, like, time_str])

            max_id = data.get("max_id", 0)
            if max_id == 0:
                print("评论已全部爬取完毕。")
                break

            delay = round(random.uniform(1.2, 2.5), 2)
            time.sleep(delay)

    print(f"爬取结束,数据已保存至：{save_path}")


if __name__ == "__main__":
    crawl_weibo_comments("QgphlzKhI", max_pages=30)
