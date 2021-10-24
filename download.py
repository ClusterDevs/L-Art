from __future__ import print_function
import os
import errno
from HTMLParser import HTMLParser
import urllib2

ARTS_LIST = 'arts-select.list'
NUMBER_TO_DOWNLOAD = 100


class MetArtHTMLParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        if tag == 'a':

            for attr in attrs:
                if (attr[0] == 'href' and
                        'selectedOrDefaultDownload' in attr[1]):
                    art_url = attr[1].split("'")[1]

                    self.data = art_url


with open(ARTS_LIST) as f:
    arts_to_download = f.readlines()
    arts_to_download = [x.strip() for x in arts_to_download]
    f.close()

myparser = MetArtHTMLParser()

for item in arts_to_download:

    pick = item.split("',")
    culture = pick[1].replace(" u'", "")
    webpage = pick[2].replace(" u'", "").replace("')", "")
    print(culture, webpage)

    response = urllib2.urlopen(webpage)
    encoding = response.headers.getparam('charset')
    html_page = response.read().decode(encoding)

    try:
        myparser.feed(html_page)

        culture = culture.replace(",", "")
        culture = culture.replace("/", " ")
        download_dir = 'data/met_art/' + culture
        try:
            os.makedirs(download_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        download_path = download_dir + '/' + myparser.data.split("/")[-1]
        image_file = open(download_path, 'wb')

        image_url = urllib2.quote(myparser.data.encode(encoding), '/:')
        print("image to download:  ", image_url)
        response = urllib2.urlopen(image_url)
        image_file.write(response.read())
        image_file.close()
    except:
        print("Error, skipping url: ", webpage)
