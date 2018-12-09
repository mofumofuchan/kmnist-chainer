#!/usr/bin/env bash
# coding: utf-8

mkdir -p data
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz -P data
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz -P data
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz -P data
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz -P data

