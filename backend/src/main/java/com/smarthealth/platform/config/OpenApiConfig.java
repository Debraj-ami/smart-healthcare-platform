package com.smarthealth.platform.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI smartHealthOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("Smart Healthcare Platform API")
                        .description("Backend Foundation API for Multi-Disease Risk Prediction and Monitoring Platform")
                        .version("1.0.0")
                        .contact(new Contact()
                                .name("Smart Healthcare Platform Team")
                                .url("https://github.com/smart-healthcare-platform"))
                        .license(new License()
                                .name("MIT")
                                .url("https://opensource.org/licenses/MIT")));
    }
}
