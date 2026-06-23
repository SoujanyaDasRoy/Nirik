"use client";

import { useEffect } from "react";

export default function FetchInterceptor() {
  useEffect(() => {
    if (typeof window === "undefined") return;

    const originalFetch = window.fetch;
    window.fetch = async function (input, init) {
      // Safe clone to prevent modifying read-only/frozen objects
      const clonedInit = init ? { ...init } : {};
      
      // Override credentials: "include" with "same-origin" to bypass CORS preflight credentials block on Hugging Face Spaces
      if (clonedInit.credentials === "include") {
        clonedInit.credentials = "same-origin";
      }

      const userStr = localStorage.getItem("nirikshon_user");
      if (userStr) {
        try {
          const user = JSON.parse(userStr);
          if (user && user.username && user.role) {
            if (clonedInit.headers instanceof Headers) {
              const newHeaders = new Headers(clonedInit.headers);
              if (!newHeaders.has("Authorization")) {
                newHeaders.set("Authorization", `Bearer ${user.username}:${user.role}`);
              }
              clonedInit.headers = newHeaders;
            } else if (Array.isArray(clonedInit.headers)) {
              const newHeaders = [...clonedInit.headers];
              const hasAuth = newHeaders.some(h => h[0].toLowerCase() === "authorization");
              if (!hasAuth) {
                newHeaders.push(["Authorization", `Bearer ${user.username}:${user.role}`]);
              }
              clonedInit.headers = newHeaders;
            } else {
              const newHeaders = clonedInit.headers ? { ...clonedInit.headers } : {};
              if (!newHeaders["Authorization"]) {
                newHeaders["Authorization"] = `Bearer ${user.username}:${user.role}`;
              }
              clonedInit.headers = newHeaders as HeadersInit;
            }
          }
        } catch (e) {
          console.error("Error patching fetch headers:", e);
        }
      }
      return originalFetch.call(this, input, clonedInit);
    };
  }, []);

  return null;
}
