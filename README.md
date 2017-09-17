# ゼロから作る Deep Learning をコピーしてきて勉強中

[オリジナルのGit](https://github.com/oreilly-japan/deep-learning-from-scratch/)  
7章のCNNを育てていきます。
1. BatichNormalization 追加 [(このバージョンを見る)](https://github.com/tom01h/deep-learning-from-scratch/tree/8e9f72143e1595a0774e939904e8c84caf0a41bf)
2. CIFAR-10 環境追加 [(このバージョンを見る)](https://github.com/tom01h/deep-learning-from-scratch/tree/3a90601683b92c5ad4bfe9ac227884183ea11b08)  
データの準備は↓
  - [ここ](https://www.cs.toronto.edu/~kriz/cifar.html)から CIFAR-10 binary version (suitable for C programs) をダウンロード
  - データを解いて ```$ tar xvzf cifar-10-binary.tar.gz```
  - データをまとめて ```$ cat cifar-10-batches-bin/data_batch_* > cifar10-train```
  - 圧縮 ```$ gzip cifar10-train```
  - test用のデータ(cifar10-test.gz)も同様にね
3. 畳み込み層2層追加
  - 畳み込み層のチャンネル数を32に全結合層のニューロン数を256に増やす
  - カーネルサイズは3に減らす

オリジナルのREADMEはここから↓

---

# ゼロから作る Deep Learning

---

![表紙](https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png)

---

本リポジトリはオライリー・ジャパン発行書籍『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』のサポートサイトです。

## ファイル構成

|フォルダ名 |説明                         |
|:--        |:--                          |
|ch01       |1章で使用するソースコード    |
|ch02       |2章で使用するソースコード    |
|...        |...                          |
|ch08       |8章で使用するソースコード    |
|common     |共通で使用するソースコード   |
|dataset    |データセット用のソースコード |


ソースコードの解説は本書籍をご覧ください。

## 必要条件
ソースコードを実行するには、下記のソフトウェアがインストールされている必要があります。

* Python 3.x
* NumPy
* Matplotlib

※Pythonのバージョンは、3系を利用します。

## 実行方法

各章のフォルダへ移動して、Pythonコマンドを実行します。

```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。

## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan＠oreilly.co.jp](<mailto:japan＠oreilly.co.jp>)までお知らせください。
