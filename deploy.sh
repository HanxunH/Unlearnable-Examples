#!/bin/bash
git checkout gh-pages-source
git branch -d gh-pages

git checkout -b gh-pages
git push --set-upstream origin gh-pages -f

git branch -d gh-pages
git checkout gh-pages-source
