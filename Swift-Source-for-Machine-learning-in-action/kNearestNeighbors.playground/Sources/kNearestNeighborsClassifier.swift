//
//  kNearestNeighborsClassifier.swift
//  Swift-Source-for-Machine-learning-in-action
//
//  Created by Robin on 27/11/2017.
//  Copyright © 2017 Robin. All rights reserved.
//

import Darwin
import Foundation

public class kNearestNeighborsClassifier {
    
    private let data:           [[Double]]
    private let labels:         [Int]
    private let nNeighbors:     Int
    
    /// 分类器构造器
    ///
    /// - Parameters:
    ///   - data: 数据集
    ///   - labels: 分类标签集
    ///   - nNeighbors: 近邻数
    public init(data: [[Double]], labels: [Int], nNeighbors: Int = 3) {
        self.data = data
        self.labels = labels
        self.nNeighbors = nNeighbors
        
        guard nNeighbors <= data.count else {
            fatalError("Expected `nNeighbors` (\(nNeighbors)) <= `data.count` (\(data.count))")
        }
        
        guard data.count == labels.count else {
            fatalError("Expected `data.count` (\(data.count)) == `labels.count` (\(labels.count))")
        }
    }
    
    /// 预测算法
    ///
    /// - Parameter xTests: 测试集
    /// - Returns: 预测分类集
    public func predict(_ xTests: [[Double]]) -> [Int] {
        return xTests.map({
            let knn = kNearestNeighbors($0)
            return kNearestNeighborsMajority(knn)
        })
    }
    
    /// 欧式距离计算
    ///
    /// - Parameters:
    ///   - xTrain: 训练集
    ///   - xTest: 测试集
    /// - Returns: 距离
    private func distance(_ xTrain: [Double], _ xTest: [Double]) -> Double {
        let distances = xTrain.enumerated().map { index, _ in
            return pow(xTrain[index] - xTest[index], 2)
        }
        
        return distances.reduce(0, +)
    }
    
    /// 距离计算&&排序
    ///
    /// - Parameter xTest: 测试集
    /// - Returns: 排序后的数据集
    private func kNearestNeighbors(_ xTest: [Double]) -> [(key: Double, value: Int)] {
        var NearestNeighbors = [Double : Int]()
        
        for (index, xTrain) in data.enumerated() {
            NearestNeighbors[distance(xTrain, xTest)] = labels[index]
        }
        
        let kNearestNeighborsSorted = Array(NearestNeighbors.sorted(by: { $0.0 < $1.0 }))[0...nNeighbors-1]
        
        return Array(kNearestNeighborsSorted)
    }
    
    /// 寻找分类标签位置
    ///
    /// - Parameter knn: 处理后的数据集
    /// - Returns: 分类标签位置
    private func kNearestNeighborsMajority(_ knn: [(key: Double, value: Int)]) -> Int {
        var labels = [Int :  Int]()
        
        for neighbor in knn {
            labels[neighbor.value] = (labels[neighbor.value] ?? 0) + 1
        }
        
        for label in labels {
            if label.value == labels.values.max() {
                return label.key
            }
        }
        
        fatalError("Cannot find the majority.")
    }
}
