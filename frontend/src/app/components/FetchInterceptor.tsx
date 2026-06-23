"use client";

import { useEffect } from "react";

export default function FetchInterceptor() {
  useEffect(() => {
    if (typeof window === "undefined") return;

    const originalFetch = window.fetch;
    window.fetch = async function (input, init) {
      const userStr = localStorage.getItem("nirikshon_user");
      if (userStr) {
        try {
          const user = JSON.parse(userStr);
          if (user && user.username && user.role) {
            init = init || {};
            init.headers = init.headers || {};
            if (init.headers instanceof Headers) {
              if (!init.headers.has("Authorization")) {
                init.headers.set("Authorization", `Bearer ${user.username}:${user.role}`);
              }
            } else if (Array.isArray(init.headers)) {
              const hasAuth = init.headers.some(h => h[0].toLowerCase() === "authorization");
              if (!hasAuth) {
                init.headers.push(["Authorization", `Bearer ${user.username}:${user.role}`]);
              }
            } else {
              if (!init.headers["Authorization"]) {
                init.headers["Authorization"] = `Bearer ${user.username}:${user.role}`;
              }
            }
          }
        } catch (e) {
          console.error("Error patching fetch headers:", e);
        }
      }
      return originalFetch.call(this, input, init);
    };
  }, []);

  return null;
}
