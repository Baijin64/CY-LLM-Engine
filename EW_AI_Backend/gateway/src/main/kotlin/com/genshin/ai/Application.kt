// Application.kt
// [启动] Spring Boot 启动类占位文件
// 说明：负责启动 Gateway 应用，初始化 Spring 上下文、相关配置与组件。

package com.genshin.ai

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.ConfigurationPropertiesScan
import org.springframework.boot.runApplication

/**
 * 网关应用启动类。
 * - 启动 Spring 上下文
 * - 扫描 `@ConfigurationProperties` 注入的配置类
 */
@SpringBootApplication
@ConfigurationPropertiesScan
class GatewayApplication

/**
 * 程序入口。使用 Spring Boot 启动应用。
 */
fun main(args: Array<String>) {
	runApplication<GatewayApplication>(*args)
}
