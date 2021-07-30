"""The code searches and saves images using the Bing API."""
# ------------------------
#   Requirements
# ------------------------
# 1. Go to the Microsoft page and create an account.
# 2. Then, go to the Bing Search API page, choose to add the Bing Search APIs v7 section.
# 3. Copy your API-key information and URL of the endpoint to the appropriate places in the code.

# ------------------------
#   USAGE
# ------------------------
# python search-bing-api.py --query "dog" --output data/dog

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
import os
import cv2
import argparse
import requests
from requests import exceptions

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="""search query to search Bing Image API for""")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

# set your Microsoft Cognitive Services API key along with:
# (1) the maximum number of results for a given search
# (2) the group size for results (maximum of 50 per request)
API_KEY = "{XXX}"
MAX_RESULTS = 1500
GROUP_SIZE = 50

# set the endpoint API Url -> must have '/v7.0/images/search'
URL = "https://api.bing.microsoft.com/v7.0/images/search"

# when attempting to download images from the web both the Python programming language and the request library
# have a number of exceptions that can be thrown so we build a list of them beforehand so we can filter on them
EXCEPTIONS = {IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError,
              exceptions.ConnectionError, exceptions.Timeout}

"""The function saves the search parameters, performs the search in Bing API and stores them."""
def start_search():
    # store the search term in a convenience variable then set the headers and search parameters
    term = args["query"]
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": term, "offset": 0, "count": GROUP_SIZE}

    # make the search
    print("[INFO] Searching Bing API for '{}'".format(term))
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()

    # grab the results from the search, including the total number of estimated results returned by the Bing API
    results = search.json()
    estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
    print("[INFO] {} total results for '{}'".format(estNumResults, term))
    search_and_save_images(estNumResults, params, headers)

"""The function accepts the search parameters ans tries for each group to download the images and save  
   them. In the event that an exception is discarded, print it and proceed to the next search result."""
def search_and_save_images(estNumResults, params, headers):
    # initialize the total number of images downloaded thus far
    total = 0

    # loop over the estimated number of results in 'GROUP_SIZE' groups
    for offset in range(0, estNumResults, GROUP_SIZE):
        sentence = "{}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults)
        # update the search parameters using the current offset, then make the request to fetch the results
        print("[INFO] Making request for group " + sentence)

        params["offset"] = offset
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print("[INFO] Saving images for group " + sentence)

        # loop over the results
        for val in results["value"]:
            try: # try to download the image
                url = val["contentUrl"]
                # make the request to download the image
                print("[INFO] Fetching: {}".format(url))
                results = requests.get(url, timeout=60)
                # build the path to the output image
                ext = url[url.rfind("."):]
                path_image = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])
                # write the image to disk
                image = open(path_image, "wb")
                image.write(results.content)
                image.close()

            # catch any errors that would not unable us to download the image
            except Exception as exception:
                # check to see if the exception is in our list of exceptions
                if type(exception) in EXCEPTIONS:
                    print("[INFO] Skipping: {}".format(url))
                    continue

            # try to load the image from disk
            img = cv2.imread(path_image)
            # update the counter
            total += 1

def main():
    start_search()
    print("The search was completed successfully!\n")

if __name__ == '__main__':
    main()