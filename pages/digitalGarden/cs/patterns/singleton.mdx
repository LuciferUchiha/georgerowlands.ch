# Singleton

The intent of the singleton pattern is that a class only has a single instance which is accessed over a global point. For example config classes should only exist once. You can do this by making the class final to stop inheritance, by making the constructor private so no instance can be created and adding a static method that creates the one instance if not already and returns it. It should either not support cloning or return the same instance(this).

## Things to be aware of

### Eager and lazy initialization

Eager means the instance is created as soon as the class is first initialized form for example other methods in the class which could use a lot of memory allthough it is not needed.

```java
public final class Singleton {
 private Singleton(){}
 private static Singleton instance = new Singleton();
 public static Singleton getInstance(){}
  return instance;
 }
}
```

Lazy means it is created when the getInstance functions is accessed this can however cause issues with multithreading which is why you need to synchronize the method.

```java
public final class Singleton {
 private Singleton(){}
 private static Singleton instance = null;
 public static synchronized Singleton getInstance(){
  if(instance == null) instance = new Singleton();
  return instance;
 }
}
```

### Garbage collection

Instance cannot be reclaimed by the garbage collector as they are static. So you should either use `WeakReference<Singelton>`(removed when not referenced by strong references) or `SoftReference<Singelton>`(removed when system is short of memory).

### Serialization

Deserialization of a serialized singleton instance may lead to several singleton instances to avoid this we can do the following

```java
public final class Singleton implements Serializable {
 private Singleton(){ }
 private static Singleton instance = null;
 public static synchronized Singleton getInstance(){
  if(instance == null) instance = new Singleton();
  return instance;
 }
 public Object readResolve(){
  return getInstance();
 }
}
```

### Singelton with Demand Holder Idiom

You can also create a Singelton the following way. This solution is thread-safe and is lazy. Important is that the construction of Singelton does not fail (exceptions).

```java
public class Singleton {
    private Singleton() {}

    private static class LazyHolder {
        static final Singleton INSTANCE = new Singleton();
    }

    public static Singleton getInstance() {
        return LazyHolder.INSTANCE;
    }
}
```

### Singelton with Enum

You can also create a Singelton with an Enum. It is easy, thread safe and provides the Serialization for free. However it can not be extended to multiple instances and the fields can not be serialized.

```java
public enum SingletonDriver implements Driver {
 INSTANCE;
 public String toString () { return "Singleton ";
 public void playSong(File file ) { ... }
}
```
