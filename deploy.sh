#!/bin/bash
git checkout gh-pages-source
git branch -d gh-pages

git checkout -b gh-pages
git push --set-upstream origin gh-pages -f

git checkout gh-pages-source
git branch -d gh-pages
